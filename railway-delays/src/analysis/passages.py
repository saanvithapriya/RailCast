"""Section passages built from OBSERVABLE data only (SIMULATED data).

A passage is one run crossing one section (two adjacent nodes u -> v of the main-line chain). Everything here uses
scheduled and actual arrival/departure times, delays, the train type, the timetable and the weather state; the
simulator's cause labels are never read.

The section increment runs from ARRIVAL at the entry node u to ARRIVAL at the exit node v, so it contains both the
running time in the section and any wait at u to enter it (queueing behind a leader, a yield, a blocked section):
    d_in  = arrival delay at u            (delay on entering the pair of stations)
    d_ent = departure delay at u          (delay at the moment the train actually enters the section)
    d_out = arrival delay at v
    delta = d_out - d_in = (d_ent - d_in) + (d_out - d_ent) = wait/halt change at u + running change in the section

Columns (minutes; t_* are absolute minutes since 2025-01-01 00:00, negative for 2024):
  run_id, train, type, dir (+1 down / -1 up), sec (section index = lower node), from_node, to_node,
  t_in (actual departure from u), t_out (actual arrival at v), d_in, d_ent, d_out, delta,
  sched_in / sched_out (scheduled times), sched_run (scheduled section time), dwell_u (scheduled dwell at u),
  weather, month, season, fast.
"""
import numpy as np
import pandas as pd

from src.graph import delay_attrs

PRIORITY_FAST = {"Rajdhani", "Duronto", "Shatabdi", "Superfast"}


def build_visits(df):
    """One row per node visit of a run (Prayagraj rows merged): first-row arrival, last-row departure."""
    vid = df.groupby("run_id", sort=False).first_in_visit.cumsum()
    g = df.assign(vid=vid.to_numpy()).groupby(["run_id", "vid"], sort=False)
    v = pd.DataFrame({
        "node": g.node.first(), "abs_arr": g.abs_arr.first(), "arr_delay": g.arrival_delay.first(), "sched_arr": g.scheduled_arrival.first(),
        "abs_dep": g.abs_dep.last(), "dep_delay": g.departure_delay.last(), "sched_dep": g.scheduled_departure.last(),
        "train": g.train.first(), "type": g.type.first(), "date": g.date.first(), "weather": g.weather.first(),
        "month": g.month.first(), "season": g.season.first(),
    }).reset_index()
    return v


def build_passages(df, bounds):
    """Passages whose arrival at v lies in the window. `df` comes from src.analysis.eda.load_window."""
    lo, hi = bounds
    v = build_visits(df)
    g = v.groupby("run_id", sort=False)
    nxt = {c: g[c].shift(-1) for c in ("node", "abs_arr", "arr_delay", "sched_arr", "weather")}
    ok = nxt["node"].notna() & ((nxt["node"] - v.node).abs() == 1) & (nxt["abs_arr"] >= lo) & (nxt["abs_arr"] < hi)
    ok &= v.abs_dep.notna() & nxt["arr_delay"].notna()
    u, w = v[ok], {c: s[ok] for c, s in nxt.items()}
    day = (pd.to_datetime(u.date) - pd.Timestamp(delay_attrs.EPOCH)).dt.days.to_numpy().astype("float64") * 1440.0
    from_node, to_node = u.node.astype("int16").to_numpy(), w["node"].astype("int16").to_numpy()
    d_ent = u.dep_delay.to_numpy().astype("float32")
    d_in = np.where(np.isnan(u.arr_delay.to_numpy()), d_ent, u.arr_delay.to_numpy()).astype("float32")      # origin visits have no arrival
    p = pd.DataFrame({
        "run_id": u.run_id.to_numpy(), "train": u.train.to_numpy(), "type": u.type.to_numpy(),
        "dir": np.where(to_node > from_node, 1, -1).astype("int8"), "sec": np.minimum(from_node, to_node),
        "from_node": from_node, "to_node": to_node, "t_in": u.abs_dep.to_numpy(), "t_out": w["abs_arr"].to_numpy(),
        "d_in": d_in, "d_ent": d_ent, "d_out": w["arr_delay"].to_numpy().astype("float32"),
        "sched_in": day + u.sched_dep.to_numpy().astype("float64"), "sched_out": day + w["sched_arr"].to_numpy().astype("float64"),
        "dwell_u": (u.sched_dep.to_numpy().astype("float64") - u.sched_arr.fillna(u.sched_dep).to_numpy().astype("float64")).astype("float32"),
        "weather": w["weather"].to_numpy(), "month": u.month.to_numpy(), "season": u.season.to_numpy(),
    })
    p["run_day"] = (day / 1440.0).astype("int32")                   # entry date of the run (days since 2025-01-01)
    p["delta"] = p.d_out - p.d_in
    p["sched_run"] = (p.sched_out - p.sched_in).astype("float32")
    p["fast"] = p.type.isin(PRIORITY_FAST).to_numpy()
    p["sched_hour"] = ((p.sched_in % 1440) // 60).astype("int8")
    p["key"] = (p.sec.astype("int32") * 2 + (p.dir > 0)).astype("int32")
    p = p.sort_values(["key", "t_in"], kind="stable").reset_index(drop=True)
    for c in ("train", "type", "weather", "season"):
        p[c] = p[c].astype("category")
    return p


def attach_leaders(p):
    """For every passage, the previous passage (by actual entry time) in the same section and direction."""
    g = p.groupby("key", sort=False)
    out = p.copy()
    for src, dst in (("t_in", "lead_t_in"), ("t_out", "lead_t_out"), ("d_ent", "lead_d_ent"), ("d_out", "lead_d_out"), ("run_id", "lead_run")):
        out[dst] = g[src].shift()
    out["lead_train"] = g["train"].shift().astype(object)
    out["gap_act"] = out.t_in - out.lead_t_in
    out["pid"] = np.arange(len(out))                       # row number in this (sorted) table
    out["lead_pid"] = out.groupby("key", sort=False).pid.shift()
    return out


def build_runs(df, tt, bounds):
    """One row per run: entry delay and time (only for runs seen from their first stop), exit delay, feeder link."""
    lo, hi = bounds
    a = df[df.first_in_visit & df.arrival_delay.notna() & (df.abs_arr >= lo) & (df.abs_arr < hi)].sort_values(["run_id", "abs_arr"])
    g = a.groupby("run_id", sort=False)
    r = pd.DataFrame({
        "train": g.train.first().astype(str), "date": g.date.first(), "type": g.type.first().astype(str),
        "first_seq": g.seq.first(), "entry_delay": g.arrival_delay.first().astype("float32"), "entry_time": g.abs_arr.first(),
        "exit_delay": g.arrival_delay.last().astype("float32"), "exit_time": g.abs_arr.last(), "n_rows": g.arrival_delay.size(),
    })
    r["day"] = ((pd.to_datetime(r.date) - pd.Timestamp(delay_attrs.EPOCH)).dt.days).astype("int32")
    spec = tt.specs
    r["entered_mid_route"] = r.train.map({t: s.entered_mid_route for t, s in spec.items()})
    r["feeder_train"] = r.train.map({t: (s.feeder[0] if s.feeder else None) for t, s in spec.items()})
    r["feeder_k"] = r.train.map({t: (s.feeder[1] if s.feeder else 0) for t, s in spec.items()})
    r["feeder_gap"] = r.train.map({t: (s.feeder[2] if s.feeder else np.nan) for t, s in spec.items()})      # scheduled turnaround (min)
    r["from_start"] = r.first_seq == 0
    return r
