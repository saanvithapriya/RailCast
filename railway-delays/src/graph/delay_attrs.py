"""Delay-based graph attributes from SIMULATED observations, computed on a chosen date range.

SIMULATED: everything here comes from the delay simulator, not from real delays.

Leakage rule: a row is used only if its ACTUAL ARRIVAL time falls inside [start 00:00, end+1 day 00:00).
The run's entry date is NOT used for the filter, because a run that enters on the last day of the window
keeps running for up to three more days; those later rows are excluded. Queue waits are filtered the same
way (by the time the wait ended). Nothing outside the window influences any number.
"""
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

EPOCH = datetime.date(2025, 1, 1)
RUN_MAX_DAYS = 4            # a run lasts up to ~3.3 days, so partitions up to 4 days before the window can matter
OBS_COLS = ["train", "date", "seq", "station", "actual_arrival", "actual_departure", "arrival_delay", "departure_delay"]


def to_day(d):
    d = pd.Timestamp(d).date()
    return (d - EPOCH).days


def window_minutes(start, end):
    """Absolute minutes (from 2025-01-01 00:00) of [start 00:00, end+1 day 00:00)."""
    return to_day(start) * 1440.0, (to_day(end) + 1) * 1440.0


def _months_between(d0, d1):
    out, cur = [], pd.Timestamp(d0).replace(day=1)
    while cur <= pd.Timestamp(d1):
        out.append(cur.strftime("%Y-%m"))
        cur = (cur + pd.offsets.MonthBegin(1))
    return out


def as_sources(scenario):
    """A scenario folder (str / Path, whole calendar allowed) or a list of splits.Source (each clipped to its range)."""
    from src.splits import Source

    if isinstance(scenario, (str, Path)):
        return [Source(str(scenario), "1900-01-01", "2999-12-31")]
    return list(scenario)


def source_bounds(src):
    """Absolute minutes [lo, hi) that a source may supply."""
    return to_day(src.start) * 1440.0, (to_day(src.end) + 1) * 1440.0


def load_queues(scenario):
    """Queue waits (SIMULATED) from every source, each clipped to the range that source may supply (by wait end)."""
    parts = []
    for s in as_sources(scenario):
        p = Path(s.dir) / "queues.parquet"
        if not p.exists():
            continue
        q = pd.read_parquet(p)
        lo, hi = source_bounds(s)
        parts.append(q[(q.wait_end >= lo) & (q.wait_end < hi)])
    if not parts:
        return pd.DataFrame(columns=["section", "direction", "wait_start", "wait_end", "train"])
    return pd.concat(parts, ignore_index=True)


def load_window_observations(scenario, start, end, columns=None):
    """Observation rows whose actual arrival or departure time lies in the window (SIMULATED).

    `scenario` is one scenario folder or a list of splits.Source (e.g. the 2024 and 2025 medium runs). Rows from a
    source are accepted only inside the calendar range that source covers, so two sources never overlap.
    Also keeps the row just before a window arrival (needed for the section it ends). Reads only the month
    partitions that can contain such rows. Adds abs_arr / abs_dep (absolute minutes).
    """
    cols = columns or OBS_COLS
    lo, hi = window_minutes(start, end)
    first = pd.Timestamp(start) - pd.Timedelta(days=RUN_MAX_DAYS)
    frames = []
    for src in as_sources(scenario):
        slo, shi = source_bounds(src)
        if shi <= lo - RUN_MAX_DAYS * 1440.0 or slo >= hi:
            continue
        parts = []
        for m in _months_between(first, end):
            p = Path(src.dir) / "observations" / f"month={m}" / "part.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p, columns=cols))
        if not parts:
            continue
        f = pd.concat(parts, ignore_index=True)
        day = ((pd.to_datetime(f.date) - pd.Timestamp(EPOCH)).dt.days).astype("float64") * 1440.0
        f["abs_arr"] = day + f.actual_arrival.astype("float64")
        f["abs_dep"] = day + f.actual_departure.astype("float64")
        ref = f.abs_arr.fillna(f.abs_dep)
        frames.append(f[(ref >= slo) & (ref < shi)])
    if not frames:
        return pd.DataFrame(columns=cols + ["abs_arr", "abs_dep"])
    df = pd.concat(frames, ignore_index=True)
    df["train"] = df.train.astype(str)
    df["station"] = df.station.astype(str)
    df = df.sort_values(["train", "date", "seq"]).reset_index(drop=True)
    nxt_arr = df.groupby(["train", "date"], sort=False).abs_arr.shift(-1)
    # keep rows that arrive or depart in the window, plus the row just before a window arrival: the upstream
    # departure delay of a passage is past information (it happened before the arrival), not leakage
    keep = ((df.abs_arr >= lo) & (df.abs_arr < hi)) | ((df.abs_dep >= lo) & (df.abs_dep < hi)) | ((nxt_arr >= lo) & (nxt_arr < hi))
    return df[keep].reset_index(drop=True)


def _prepare(obs, node_of):
    """Add node index, first-row-of-visit flag, previous node and previous departure delay per run."""
    df = obs.sort_values(["train", "date", "seq"]).reset_index(drop=True)
    df["node"] = df.station.map(node_of)
    g = df.groupby(["train", "date"], sort=False)
    df["prev_node"] = g.node.shift()
    df["prev_dep_delay"] = g.departure_delay.shift()
    df["first_in_visit"] = df.node != df.prev_node
    return df


def _in_window(df, start, end):
    lo, hi = window_minutes(start, end)
    return df[(df.abs_arr >= lo) & (df.abs_arr < hi)]


def node_delay_attributes(scenario_dir, start, end, node_of):
    """SIMULATED arrival-delay statistics per node (one arrival per visit; PRYG counted once) in the window."""
    obs = load_window_observations(scenario_dir, start, end)
    df = _in_window(_prepare(obs, node_of), start, end)
    df = df[df.first_in_visit & df.arrival_delay.notna()]
    g = df.groupby("node").arrival_delay
    out = pd.DataFrame({
        "n_arrivals": g.size(), "mean_arrival_delay": g.mean(), "median_arrival_delay": g.median(),
        "p90_arrival_delay": g.quantile(0.9), "share_late_over_10": g.apply(lambda s: (s > 10).mean()),
    }).reset_index()
    out["window_start"], out["window_end"], out["is_simulated"] = str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()), True
    return out


def edge_delay_attributes(scenario_dir, start, end, node_of, queues=None):
    """SIMULATED delay statistics per directed edge (from_node -> to_node) in the window.

    * arrival-delay mean / median / p90 at the downstream node, over passages whose arrival is in the window
    * mean_delay_added = mean of (arrival delay downstream - departure delay upstream)
    * queue: mean wait before entering the section per passage, and share of passages that waited
      (only waits longer than 0.5 minute are logged)
    Off-corridor legs and multi-section hops (skipped rows) are not edges and are excluded.
    """
    obs = load_window_observations(scenario_dir, start, end)
    df = _in_window(_prepare(obs, node_of), start, end)
    df = df[df.first_in_visit & df.arrival_delay.notna() & df.prev_node.notna() & df.prev_dep_delay.notna()]
    df = df[(df.node - df.prev_node).abs() == 1].copy()
    df["added"] = df.arrival_delay.astype("float64") - df.prev_dep_delay.astype("float64")
    df["from_node"], df["to_node"] = df.prev_node.astype(int), df.node.astype(int)
    g = df.groupby(["from_node", "to_node"])
    out = pd.DataFrame({
        "n_passages": g.size(), "mean_arrival_delay": g.arrival_delay.mean(), "median_arrival_delay": g.arrival_delay.median(),
        "p90_arrival_delay": g.arrival_delay.quantile(0.9), "mean_delay_added": g.added.mean(),
    }).reset_index()
    if queues is None:
        queues = load_queues(scenario_dir)
    lo, hi = window_minutes(start, end)
    q = queues[(queues.wait_end >= lo) & (queues.wait_end < hi)].copy()
    q["wait"] = q.wait_end - q.wait_start
    q["from_node"] = np.where(q.direction == 1, q.section, q.section + 1)
    q["to_node"] = np.where(q.direction == 1, q.section + 1, q.section)
    qa = q.groupby(["from_node", "to_node"]).wait.agg(["size", "sum"]).rename(columns={"size": "n_waits", "sum": "wait_minutes"}).reset_index()
    out = out.merge(qa, on=["from_node", "to_node"], how="left").fillna({"n_waits": 0, "wait_minutes": 0})
    out["mean_queue_wait_per_passage"] = out.wait_minutes / out.n_passages
    out["share_passages_waited"] = out.n_waits / out.n_passages
    out["window_start"], out["window_end"], out["is_simulated"] = str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()), True
    return out


# ----------------------------------------------------------------------------- windows named in configs/splits.yaml
def node_delay_for_window(name, node_of, scenario="medium"):
    """Node delay attributes for a named window of configs/splits.yaml ('train', 'validation', 'test', 'analysis_all_2025')."""
    from src import splits

    a, b = splits.window(name)
    return node_delay_attributes(splits.sources(scenario), a, b, node_of)


def edge_delay_for_window(name, node_of, scenario="medium"):
    from src import splits

    a, b = splits.window(name)
    return edge_delay_attributes(splits.sources(scenario), a, b, node_of)


def fold_train_delay_tables(fold_name, node_of, scenario="medium", part="train", sources=None):
    """Node and edge delay tables built ONLY from one rolling-origin fold's training range (or its validation range).

    For fold 1 this uses data up to 2024-10-31 only (by actual event time), so nothing from the validation
    fold's Nov-Dec 2024 data enters. Returns (node_table, edge_table), SIMULATED.
    """
    from src import splits

    a, b = splits.fold(fold_name)[part]
    src = sources if sources is not None else splits.sources(scenario)
    return node_delay_attributes(src, a, b, node_of), edge_delay_attributes(src, a, b, node_of)
