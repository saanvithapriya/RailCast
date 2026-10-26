"""Ground-truth cascades from the simulator's cause labels (SIMULATED, VALIDATION ONLY).

*** No mining or prediction method may read these files. ***
They exist only to check, in Phase 5, whether mining recovers the cascades the simulator actually produced.
They are built from delay_attribution (the simulator's labels), which a real dataset would not have.

A cascade is everything rooted at one injected event (`root_event_id`): the set of train runs whose delay
contains minutes that trace back to that event, linked by the "caused by train X" labels.
* hop 0: runs hit directly by the event (incident / halt extension / entry delay on their own train, or held
  by the disruption / freight blocker itself)
* hop h+1: runs that waited for, yielded to, or inherited a rake delay from a run at hop h
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import delay_attrs
from src.simulation import causes as C

DIRECT = (C.INCIDENT, C.DISRUPTION, C.FREIGHT, C.HALT_EXT, C.ENTRY)
BY_TRAIN = (C.SECTION_WAIT, C.YIELD, C.RAKE)
CASCADES_FILE = "groundtruth_cascades.parquet"
RUNS_FILE = "groundtruth_cascade_runs.parquet"
NOTICE = "SIMULATED. VALIDATION ONLY: no mining or prediction method may read this file (it uses simulator labels)."


def _collect(scenario_dir):
    scen = Path(scenario_dir)
    months = sorted(p.name.split("=")[1] for p in (scen / "observations").glob("month=*"))
    run_parts, edge_parts, direct_parts = [], [], []
    for m in months:
        obs = pd.read_parquet(scen / "observations" / f"month={m}" / "part.parquet",
                              columns=["obs_id", "train", "date", "station", "seq", "actual_arrival"])
        at = pd.read_parquet(scen / "delay_attribution" / f"month={m}" / "part.parquet")
        at = at[(at.phase == "arr") & (at.root_event_id > 0) & (at.cause_code > 0)]
        a = at.merge(obs, on="obs_id")
        a["train"], a["cause_train"] = a.train.astype(str), a.cause_train.astype(str).replace("nan", "")
        a["day"] = ((pd.to_datetime(a.date) - pd.Timestamp(delay_attrs.EPOCH)).dt.days).astype(int)
        a["t"] = a.day * 1440.0 + a.actual_arrival.astype("float64")
        g = a.groupby(["root_event_id", "train", "day"])
        run_parts.append(pd.DataFrame({
            "first_time": g.t.min(), "last_time": g.t.max(), "new_minutes": g.new_minutes.sum(),
            "peak_minutes": g.minutes.max()}).reset_index())
        e = a[a.cause_code.isin(BY_TRAIN) & (a.new_minutes >= 0.5)]
        edge_parts.append(e[["root_event_id", "train", "day", "cause_train", "cause_code", "station", "seq", "new_minutes", "t"]])
        d = a[a.cause_code.isin(DIRECT) & (a.event_id == a.root_event_id)]
        direct_parts.append(d[["root_event_id", "train", "day"]].drop_duplicates())
    runs = pd.concat(run_parts, ignore_index=True)
    runs = runs.groupby(["root_event_id", "train", "day"]).agg(
        first_time=("first_time", "min"), last_time=("last_time", "max"), new_minutes=("new_minutes", "sum"),
        peak_minutes=("peak_minutes", "max")).reset_index()
    return runs, pd.concat(edge_parts, ignore_index=True), pd.concat(direct_parts, ignore_index=True).drop_duplicates()


def _hops(run_keys, direct_keys, edges):
    """Hop count per run by breadth-first search over 'caused by train X' links within one cascade."""
    by_train = {}
    for k in run_keys:
        by_train.setdefault(k[0], []).append(k[1])
    rk = set(run_keys)
    hop = {k: 0 for k in direct_keys if k in rk}
    changed = True
    rows = list(edges.itertuples(index=False))
    rows.sort(key=lambda r: r.t)
    while changed:
        changed = False
        for r in rows:
            tgt = (r.train, r.day)
            days = by_train.get(r.cause_train)
            if not days:
                continue
            src = (r.cause_train, min(days, key=lambda d: (abs(d - r.day), d)))
            if src in hop and src != tgt and (tgt not in hop or hop[tgt] > hop[src] + 1):
                hop[tgt] = hop[src] + 1
                changed = True
    return hop


def build_groundtruth_cascades(scenario_dir, out_dir=None):
    """Build and save the two ground-truth tables next to the scenario output. Returns (cascades, runs)."""
    scen = Path(scenario_dir)
    out = Path(out_dir or scen)
    events = pd.read_parquet(scen / "events.parquet").set_index("event_id")
    runs, edges, direct = _collect(scen)
    edge_groups = {k: g for k, g in edges.groupby("root_event_id")}
    direct_groups = {k: set(zip(g.train, g.day)) for k, g in direct.groupby("root_event_id")}
    run_groups = {k: g for k, g in runs.groupby("root_event_id")}
    casc, runrows = [], []
    for root, g in run_groups.items():
        if root not in events.index:
            continue
        ev = events.loc[root]
        keys = list(zip(g.train, g.day))
        e = edge_groups.get(root, edges.iloc[0:0])
        hop = _hops(keys, direct_groups.get(root, set()), e)
        g = g.assign(hop=[hop.get(k, np.nan) for k in keys])
        n_unlinked = int(g.hop.isna().sum())
        casc.append({
            "origin_event_id": int(root), "origin_event_type": str(ev.type), "origin_place": ev.place,
            "origin_train": ev.train, "origin_date": pd.Timestamp(ev.date), "origin_start_min": float(ev.start_min),
            "origin_injected_min": float(ev.duration_min), "size_runs_delayed": int(len(g)),
            "n_trains_delayed": int(g.train.nunique()), "max_depth_hops": float(g.hop.max()) if g.hop.notna().any() else np.nan,
            "n_runs_without_hop": n_unlinked, "duration_min": float(g.last_time.max() - ev.start_min),
            "total_delay_minutes_added": float(g.new_minutes.sum()),
            "first_delay_time_min": float(g.first_time.min()), "last_delay_time_min": float(g.last_time.max()),
        })
        g = g.assign(origin_event_id=int(root))
        runrows.append(g)
    cascades = pd.DataFrame(casc).sort_values("origin_start_min").reset_index(drop=True)
    cascades.insert(0, "cascade_id", np.arange(len(cascades)))
    cid = dict(zip(cascades.origin_event_id, cascades.cascade_id))
    runs_out = pd.concat(runrows, ignore_index=True)
    runs_out["cascade_id"] = runs_out.origin_event_id.map(cid)
    runs_out = runs_out.rename(columns={"day": "run_day", "new_minutes": "delay_minutes_added", "peak_minutes": "peak_delay_minutes_from_event"})
    runs_out = runs_out[["cascade_id", "origin_event_id", "train", "run_day", "hop", "first_time", "last_time",
                         "delay_minutes_added", "peak_delay_minutes_from_event"]].sort_values(["cascade_id", "hop", "first_time"])
    for df in (cascades, runs_out):
        df["is_simulated"] = True
        df["validation_only"] = True
    out.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    for df, name in ((cascades, CASCADES_FILE), (runs_out, RUNS_FILE)):
        t = pa.Table.from_pandas(df, preserve_index=False)
        t = t.replace_schema_metadata({**(t.schema.metadata or {}), b"SIMULATED": b"true", b"VALIDATION_ONLY": NOTICE.encode()})
        pq.write_table(t, out / name, compression="zstd")
    (out / "README_GROUNDTRUTH_VALIDATION_ONLY.txt").write_text(
        NOTICE + "\n\nWhat it is: cascades rooted at injected events, from the simulator's cause labels.\n"
        "Who may read it: only Phase 5 validation code, to compare against mined cascades.\n"
        "Never use it to build graphs, features or labels for mining or prediction.\n")
    return cascades, runs_out


def load_groundtruth(scenario_dir, *, validation_only=False):
    """Load the ground-truth tables. Refuses unless the caller states this is validation code."""
    if not validation_only:
        raise PermissionError("groundtruth cascades are for validation only; pass validation_only=True from Phase 5 validation code")
    p = Path(scenario_dir)
    return pd.read_parquet(p / CASCADES_FILE), pd.read_parquet(p / RUNS_FILE)
