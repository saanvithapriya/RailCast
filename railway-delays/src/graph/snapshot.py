"""Time-expanded snapshot (SIMULATED delays): which trains are at each station, or in the section leading to
it, at a given date and hour, and what their current delay is. No features are built here.

Position at time t (absolute minutes) uses only events with times <= t, plus the timetable:
* at_station:  the train arrived at the node at or before t and has not left yet; delay = its arrival delay
* in_section:  the train left the previous node before t and has not arrived yet; it is attached to the
               downstream node (its next station) and delay = its departure delay at the previous node
* Trains in an off-corridor leg (the 39 trains that leave the main line) are not attached to any node.
`scheduled_*` columns are the timetable times at that node (for in_section: at the downstream node).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .delay_attrs import EPOCH, RUN_MAX_DAYS, _months_between, as_sources, source_bounds, to_day

SNAP_COLS = ["train", "date", "seq", "station", "scheduled_arrival", "scheduled_departure", "actual_arrival",
             "actual_departure", "arrival_delay", "departure_delay"]


def load_runs_around(scenario_dir, t_abs):
    """Observation rows of runs that could be active at absolute time t (entered within the last few days).

    `scenario_dir` is a folder or a list of splits.Source (rows are clipped to each source's calendar range)."""
    day = int(t_abs // 1440)
    first = pd.Timestamp(EPOCH) + pd.Timedelta(days=day - RUN_MAX_DAYS)
    last = pd.Timestamp(EPOCH) + pd.Timedelta(days=day)
    frames = []
    for src in as_sources(scenario_dir):
        slo, shi = source_bounds(src)
        for m in _months_between(first, last):
            p = Path(src.dir) / "observations" / f"month={m}" / "part.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p, columns=SNAP_COLS)
            df = df[(pd.to_datetime(df.date) >= first) & (pd.to_datetime(df.date) <= last)].copy()
            d0 = ((pd.to_datetime(df.date) - pd.Timestamp(EPOCH)).dt.days).astype("float64") * 1440.0
            for c in ("scheduled_arrival", "scheduled_departure", "actual_arrival", "actual_departure"):
                df["abs_" + c.split("_")[0][:3] + "_" + c.split("_")[1][:3]] = d0 + df[c].astype("float64")
            ref = df.abs_act_arr.fillna(df.abs_act_dep)
            frames.append(df[(ref >= slo) & (ref < shi)])
    df = pd.concat(frames, ignore_index=True)
    df["train"], df["station"] = df.train.astype(str), df.station.astype(str)
    return df


def positions_at(rows, t_abs, node_of):
    """Pure function: where is every run at absolute time t? `rows` as returned by load_runs_around."""
    df = rows.sort_values(["train", "date", "seq"]).reset_index(drop=True)
    df["node"] = df.station.map(node_of)
    arr = df.abs_act_arr.fillna(df.abs_act_dep)
    dep = df.abs_act_dep.fillna(df.abs_act_arr)
    g = df.groupby(["train", "date"], sort=False)
    nxt_node = g.node.shift(-1)
    nxt_arr = arr.groupby([df.train, df.date], sort=False).shift(-1)
    out = []

    here = (arr <= t_abs) & (dep >= t_abs)
    h = df[here]
    out.append(pd.DataFrame({
        "node": h.node.astype(int), "train": h.train, "run_date": h.date, "state": "at_station", "from_node": -1,
        "station": h.station, "current_delay_min": h.arrival_delay.fillna(h.departure_delay).astype("float64"),
        "last_event_min": arr[here],
        "scheduled_arrival_min": h.abs_sch_arr, "scheduled_departure_min": h.abs_sch_dep,
    }))
    sec = (dep < t_abs) & (nxt_arr > t_abs) & nxt_node.notna()
    s = df[sec]
    nn = nxt_node[sec].astype(int)
    adj = (nn - s.node).abs() == 1
    s, nn = s[adj], nn[adj]
    nxt_rows = df.shift(-1).loc[s.index]
    out.append(pd.DataFrame({
        "node": nn, "train": s.train, "run_date": s.date, "state": "in_section", "from_node": s.node.astype(int),
        "station": nxt_rows.station, "current_delay_min": s.departure_delay.fillna(s.arrival_delay).astype("float64"),
        "last_event_min": dep[s.index],
        "scheduled_arrival_min": nxt_rows.abs_sch_arr, "scheduled_departure_min": nxt_rows.abs_sch_dep,
    }))
    res = pd.concat(out, ignore_index=True).sort_values(["node", "state", "train"]).reset_index(drop=True)
    res["is_simulated"] = True
    return res


def snapshot(scenario_dir, date, hour, tt, minute=0):
    """Snapshot at date + hour:minute for every node. SIMULATED delays; returns one row per (train, node)."""
    t_abs = to_day(date) * 1440.0 + hour * 60.0 + minute
    rows = load_runs_around(scenario_dir, t_abs)
    snap = positions_at(rows, t_abs, tt.node_of)
    snap["node_name"] = snap.node.map(dict(enumerate(tt.net.node_names)))
    snap["type"] = snap.train.map({t: s.type for t, s in tt.specs.items()})
    snap["time_abs_min"] = t_abs
    return snap
