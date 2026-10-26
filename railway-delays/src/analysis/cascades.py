"""Item 3: cascade extraction from OBSERVABLE data only (SIMULATED data).

Every run gets at most one cause, the earliest of:
  rake     the run enters the corridor late (>= R_min) and the run that feeds its rake (timetable link) left the corridor
           at least as late (>= R_min, and not less than the entry delay minus one minute), and the scheduled turnaround
           between them is at most G_rake minutes (beyond that the entry delay no longer follows the feeder delay)
  external the run enters the corridor at least T_entry minutes late and no feeder explains it
  primary  the run has a primary event: a passage adding >= P minutes that no late leader explains (item 1)
  link     the run adds >= A_link minutes in a section and is explained by a late leader (item 1): its parent is that leader
(ties are broken in the order above). A run belongs to a cascade if its chain of parents ends in a run whose cause is
external or primary (the trigger); runs whose chain ends in a late leader with no identified cause are not in any
cascade. Cascades are therefore rooted trees of runs.

Stored per cascade: id, origin type / section / time / run, size (runs), depth (hops), duration (minutes from the origin
to the last run affected), spatial extent (distinct sections where members were hit), total delay minutes, weather and
month of the origin.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from . import decomposition as dc

NONE, RAKE, EXT, PRIM, LINK = 0, 1, 2, 3, 4
CAUSE_NAMES = {NONE: "none", RAKE: "rake", EXT: "external_entry", PRIM: "primary", LINK: "section_link"}


def extract_cascades(p, runs, prm, explained=None, primary=None, max_iter=80):
    """Return (cascades, members, links). `p`: decomposed passages with leaders; `runs`: passages.build_runs output."""
    if explained is None:
        explained = dc.flag_explained(p, prm)
    if primary is None:
        primary = (p.added >= prm.P) & ~explained
    R = runs.reset_index()
    n = len(R)
    ridx = pd.Series(np.arange(n), index=R.run_id.to_numpy())
    inf = np.inf

    # first section link and first primary event per run
    lk = p[explained & (p.added >= prm.A_link) & p.lead_run.notna()].sort_values("t_out").drop_duplicates("run_id")
    link_time = np.full(n, inf); link_parent = np.full(n, -1); link_sec = np.full(n, -1); link_added = np.zeros(n)
    i = ridx.reindex(lk.run_id.to_numpy()).to_numpy()
    link_time[i], link_sec[i], link_added[i] = lk.t_out.to_numpy(), lk.sec.to_numpy(), lk.added.to_numpy()
    par = ridx.reindex(lk.lead_run.to_numpy().astype("int64")).to_numpy()
    link_parent[i] = np.where(np.isnan(par), -1, par).astype(int)

    pe = p[primary].sort_values("t_in").drop_duplicates("run_id")
    prim_time = np.full(n, inf); prim_sec = np.full(n, -1); prim_added = np.zeros(n)
    i = ridx.reindex(pe.run_id.to_numpy()).to_numpy()
    prim_time[i], prim_sec[i], prim_added[i] = pe.t_in.to_numpy(), pe.sec.to_numpy(), pe.added.to_numpy()

    first = p.sort_values("t_in").drop_duplicates("run_id")
    first_sec = np.full(n, -1); first_wx = np.full(n, "", dtype=object)
    i = ridx.reindex(first.run_id.to_numpy()).to_numpy()
    first_sec[i] = first.sec.to_numpy(); first_wx[i] = first.weather.astype(str).to_numpy()

    # rake feeder (timetable link) and external entry
    fkey = pd.MultiIndex.from_arrays([R.feeder_train.fillna("").to_numpy(), (R.day - R.feeder_k).to_numpy()])
    feeder = pd.Series(np.arange(n), index=pd.MultiIndex.from_arrays([R.train.to_numpy(), R.day.to_numpy()])).reindex(fkey).to_numpy()
    has_feeder = ~np.isnan(feeder)
    feeder_i = np.where(has_feeder, feeder, 0).astype(int)
    f_exit = np.where(has_feeder, R.exit_delay.to_numpy()[feeder_i], np.nan)
    entry, e_time, start = R.entry_delay.to_numpy("float64"), R.entry_time.to_numpy("float64"), R.from_start.to_numpy()
    gap = R.feeder_gap.to_numpy("float64") if "feeder_gap" in R else np.zeros(n)
    rake_ok = start & has_feeder & (entry >= prm.R_min) & (f_exit >= prm.R_min) & (entry <= f_exit + 1.0) & (gap <= prm.G_rake)
    ext_ok = start & (entry >= prm.T_entry) & ~rake_ok

    times = np.column_stack([np.where(rake_ok, e_time, inf), np.where(ext_ok, e_time, inf), prim_time, link_time])
    cause = np.where(np.isinf(times).all(axis=1), NONE, np.argmin(times, axis=1) + 1)
    cause_time = np.where(cause == NONE, np.nan, times[np.arange(n), np.maximum(cause - 1, 0)])
    parent = np.full(n, -1)
    parent = np.where(cause == RAKE, feeder_i, parent)
    parent = np.where(cause == LINK, link_parent, parent)
    cause_sec = np.select([cause == PRIM, cause == LINK], [prim_sec, link_sec], default=first_sec)
    cause_amt = np.select([cause == PRIM, cause == LINK, cause == RAKE, cause == EXT], [prim_added, link_added, entry, entry], default=0.0)

    # membership: roots (external / primary) and runs whose chain of parents reaches a root
    member = (cause == EXT) | (cause == PRIM)
    root = np.where(member, np.arange(n), -1)
    depth = np.zeros(n, dtype=int)
    child = ((cause == RAKE) | (cause == LINK)) & (parent >= 0)
    psafe = np.where(parent >= 0, parent, 0)
    for _ in range(max_iter):
        newly = child & ~member & member[psafe]
        if not newly.any():
            break
        member[newly] = True
        root[newly] = root[parent[newly]]
        depth[newly] = depth[parent[newly]] + 1

    m = pd.DataFrame({"run_idx": np.arange(n), "run_id": R.run_id.to_numpy(), "train": R.train.to_numpy(), "day": R.day.to_numpy(),
                      "root": root, "depth": depth, "cause": cause, "cause_name": pd.Series(cause).map(CAUSE_NAMES).to_numpy(),
                      "cause_time": cause_time, "cause_sec": cause_sec, "amount": cause_amt, "parent": parent})
    m = m[member].copy()
    g = m.groupby("root")
    casc = pd.DataFrame({
        "size_runs": g.size(), "depth": g.depth.max(), "extent_sections": g.cause_sec.nunique(),
        "total_delay_min": g.amount.sum(), "last_time": g.cause_time.max(),
    })
    ro = m.set_index("run_idx").loc[casc.index]
    casc["origin_type"] = ro.cause_name.to_numpy()
    casc["origin_sec"], casc["origin_time"] = ro.cause_sec.to_numpy(), ro.cause_time.to_numpy()
    casc["origin_run_id"], casc["origin_train"], casc["origin_day"] = ro.run_id.to_numpy(), ro.train.to_numpy(), ro.day.to_numpy()
    casc["origin_amount"] = ro.amount.to_numpy()
    casc["origin_weather"] = first_wx[casc.index.to_numpy()]
    casc["duration_min"] = casc.last_time - casc.origin_time
    casc["origin_month"] = (pd.Timestamp("2025-01-01") + pd.to_timedelta(casc.origin_time, unit="m")).dt.month.to_numpy()
    casc = casc.reset_index().rename(columns={"root": "root_run_idx"})
    casc.insert(0, "cascade_id", np.arange(len(casc)))
    cid = pd.Series(casc.cascade_id.to_numpy(), index=casc.root_run_idx.to_numpy())
    m["cascade_id"] = cid.reindex(m.root.to_numpy()).to_numpy()
    links = m[m.parent >= 0][["cascade_id", "run_idx", "parent", "cause", "cause_sec", "cause_time", "amount", "depth"]].rename(
        columns={"run_idx": "child", "cause_sec": "sec", "cause_time": "time"})
    links["parent_run_id"] = R.run_id.to_numpy()[links.parent.to_numpy()]
    links["child_run_id"] = R.run_id.to_numpy()[links.child.to_numpy()]
    return casc, m, links


def summarise(casc):
    """Headline numbers of a cascade table."""
    s = casc.size_runs
    return {"cascades": int(len(casc)), "share_size_2plus": float((s >= 2).mean()), "median_size": float(s.median()),
            "p90_size": float(s.quantile(0.9)), "max_size": int(s.max()), "mean_size": float(s.mean()),
            "share_depth_2plus": float((casc.depth >= 2).mean()), "max_depth": int(casc.depth.max()),
            "median_duration_min": float(casc.duration_min.median()), "primary": int((casc.origin_type == "primary").sum()),
            "external_entry": int((casc.origin_type == "external_entry").sum())}


def sensitivity(p, runs, prm, p_values=(10, 15, 20, 30), g_values=(4, 6, 10, 15)):
    """Cascade summaries for different primary thresholds P (G fixed) and gap windows G (P fixed)."""
    rows = []
    for P in p_values:
        c, _, _ = extract_cascades(p, runs, replace(prm, P=float(P), T_entry=float(P)))
        rows.append({"varied": "P and entry trigger", "value": P, **summarise(c)})
    for G in g_values:
        c, _, _ = extract_cascades(p, runs, replace(prm, G=float(G)))
        rows.append({"varied": "G", "value": G, **summarise(c)})
    return pd.DataFrame(rows)
