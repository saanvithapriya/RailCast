"""Phase 4 exploratory analysis helpers (SIMULATED data, medium scenario).

Rules enforced here:
* Windows come from configs/splits.yaml (via src/splits.py). Analyses that inform modelling decisions use only
  the 'decision' window (2024-01-01 .. 2024-10-31, fold 1's training range); the rest of the training window
  (2024-11-01 .. 2025-08-31) is used for descriptive plots only. Validation fold data after the training window
  and the test window (Nov-Dec 2025) are never loaded.
* Only what real data would contain is read: times and arrival/departure delays, plus the weather state and the
  timetable. The cause labels in the observation table (cause_code, cause_train, root_event_id,
  primary_delay_injected), the delay_attribution table, the events table, the queue log and the ground-truth
  cascades are never loaded.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import splits
from src.graph import delay_attrs
from src.graph.station_graph import build_station_tables
from src.graph.timetable import load_timetable

OBS_COLS = ["train", "date", "seq", "station", "scheduled_arrival", "scheduled_departure", "actual_arrival",
            "actual_departure", "arrival_delay", "departure_delay"]          # observable columns only
SHATABDI_ARTIFACT = {"12003", "12033"}                                       # do not interpret per-train results
FOG_SEASON = "Jan-Feb (fog season)"
SEASONS = [FOG_SEASON, "Mar-May", "Jun-Aug", "Sep-Oct", "Nov-Dec (fog season)"]


def season_of(month):
    m = np.asarray(month)
    return np.select([m <= 2, m <= 5, m <= 8, m <= 10], SEASONS[:4], default=SEASONS[4])


def load_window(start, end, scenario="medium", tt=None, nodes=None):
    """Observation rows with actual arrival/departure in [start, end] (SIMULATED) plus derived columns.

    Returns (df, tt, nodes, (lo, hi)) with lo/hi the absolute-minute bounds of the window.
    """
    tt = tt or load_timetable()
    nodes = nodes if nodes is not None else build_station_tables(tt)[0]
    sources = splits.sources(scenario)
    lo, hi = delay_attrs.window_minutes(start, end)
    obs = delay_attrs.load_window_observations(sources, start, end, columns=OBS_COLS)
    df = obs.sort_values(["train", "date", "seq"]).reset_index(drop=True)
    df["run_id"] = df.groupby(["train", "date"], sort=False).ngroup().astype("int32")
    df["node"] = df.station.map(tt.node_of).astype("int16")
    g = df.groupby("run_id", sort=False)
    df["prev_node"] = g.node.shift()
    df["prev_dep_delay"] = g.departure_delay.shift()
    df["first_in_visit"] = df.node != df.prev_node
    spec_dir = {t: int(np.sign(sum(h.direction for h in sp.hops if h.kind == "section"))) for t, sp in tt.specs.items()}
    df["direction"] = df.train.map(spec_dir).astype("int8")
    df["type"] = df.train.map({t: s.type for t, s in tt.specs.items()})
    df["sched_hour"] = ((df.scheduled_arrival.astype("float64") % 1440) // 60).astype("int8")
    df["H"] = ((df.abs_arr - lo) // 60)                                            # hours since the window start
    ts = pd.Timestamp(delay_attrs.EPOCH) + pd.to_timedelta((df.abs_arr // 1440).fillna(0), unit="D")
    df["arr_date"] = ts
    df["month"] = ts.dt.month.astype("int8")
    df["dow"] = ts.dt.dayofweek.astype("int8")
    wx = pd.concat([pd.read_parquet(Path(s.dir) / "weather.parquet") for s in sources], ignore_index=True)
    wx["date"] = pd.to_datetime(wx.date)
    wx = wx[(wx.date >= pd.Timestamp(start)) & (wx.date <= pd.Timestamp(end))]          # weather of the window only
    df["weather"] = df.arr_date.map(dict(zip(wx.date, wx.state.astype(str))))
    df["season"] = season_of(df.month.to_numpy())
    for c in ("train", "station", "type", "weather", "season"):
        df[c] = df[c].astype("category")
    return df, tt, nodes, (lo, hi)


def arrivals(df, bounds):
    """One arrival per node visit (Prayagraj counted once) inside the window, with a delay value."""
    lo, hi = bounds
    a = df[df.first_in_visit & df.arrival_delay.notna() & (df.abs_arr >= lo) & (df.abs_arr < hi)].copy()
    a["arrival_delay"] = a.arrival_delay.astype("float32")
    return a


# ----------------------------------------------------------------------------- node-hour arrays
def node_hour_arrays(a, n_nodes, n_hours):
    """mean arrival delay and arrival count per (node, direction, hour since window start). Direction 0=down, 1=up."""
    d = a.assign(di=(a.direction < 0).astype(int), Hi=a.H.astype(int))
    d = d[(d.Hi >= 0) & (d.Hi < n_hours)]
    g = d.groupby(["node", "di", "Hi"], observed=True).arrival_delay.agg(["mean", "size"]).reset_index()
    mean = np.full((n_nodes, 2, n_hours), np.nan, dtype="float32")
    cnt = np.zeros((n_nodes, 2, n_hours), dtype="int16")
    mean[g.node, g.di, g.Hi] = g["mean"]
    cnt[g.node, g.di, g.Hi] = g["size"]
    return mean, cnt


def interpolate_gaps(x, limit=3):
    """Linear interpolation over gaps of at most `limit` hours along the last axis."""
    out = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = pd.Series(x[i, j]).interpolate(limit=limit, limit_area="inside").to_numpy()
    return out


def disturbance_episodes(series, high=30.0, low=15.0):
    """Hysteresis episodes: onset when value >= high, recovery at the first value < low.

    Returns (onset_hour, duration_hours, recovered) per episode; an episode still open at the end is censored."""
    eps = []
    on, start = False, 0
    for t, v in enumerate(series):
        if np.isnan(v):
            continue
        if not on and v >= high:
            on, start = True, t
        elif on and v < low:
            eps.append((start, t - start, True))
            on = False
    if on:
        eps.append((start, len(series) - start, False))
    return eps


def kaplan_meier(durations, recovered):
    """Kaplan-Meier survival of 'still disturbed' (event = recovery). Returns DataFrame(time, survival, at_risk)."""
    d = pd.DataFrame({"t": np.asarray(durations), "e": np.asarray(recovered).astype(int)})
    s, rows = 1.0, [(0, 1.0, len(d))]
    for t in np.sort(d.t.unique()):
        n = int((d.t >= t).sum())
        ev = int(((d.t == t) & (d.e == 1)).sum())
        if n > 0:
            s *= 1 - ev / n
        rows.append((t, s, n))
    return pd.DataFrame(rows, columns=["time", "survival", "at_risk"])


def acf(x, max_lag):
    """Autocorrelation of a series with gaps (NaN): correlation over the pairs where both values exist."""
    x = pd.Series(x)
    return np.array([x.corr(x.shift(k)) for k in range(1, max_lag + 1)])


# ----------------------------------------------------------------------------- pairs
def same_node_pairs(a, resid=None):
    """Consecutive arrivals of different trains at the same node and direction: gap (min) and both delays."""
    d = a.sort_values(["node", "direction", "abs_arr"])
    key = (d.node.astype(int) * 4 + (d.direction < 0).astype(int)).to_numpy()
    prev_delay = d.arrival_delay.groupby(key).shift()
    prev_t = d.abs_arr.groupby(key).shift()
    prev_train = d.train.astype(str).groupby(key).shift()
    out = pd.DataFrame({"gap": d.abs_arr - prev_t, "delay": d.arrival_delay, "prev_delay": prev_delay,
                        "same_train": d.train.astype(str) == prev_train, "node": d.node})
    if resid is not None:
        r = resid.loc[d.index]
        out["resid"] = r
        out["prev_resid"] = r.groupby(key).shift()
    return out.dropna(subset=["gap", "prev_delay"])


def residualise(a):
    """Delay minus its (node, direction, hour-of-day, month) mean: removes the shared daily/seasonal pattern."""
    k = [a.node, a.direction, a.sched_hour, a.month]
    return a.arrival_delay - a.groupby(k, observed=True).arrival_delay.transform("mean")


# ----------------------------------------------------------------------------- train-level horizon pairs
def horizon_pairs(a, h, tol_h=0.5):
    """For every arrival (the 'now' row of a run), the arrival of the same run about h hours later."""
    d = a.sort_values(["run_id", "abs_arr"])
    key = d.run_id.to_numpy(dtype="float64") * 1e7 + d.abs_arr.to_numpy(dtype="float64")
    run = d.run_id.to_numpy()
    t = d.abs_arr.to_numpy(dtype="float64")
    idx = np.searchsorted(key, key + 60.0 * h)
    best = np.full(len(d), -1)
    bestd = np.full(len(d), np.inf)
    for c in (idx - 1, idx):
        cc = np.clip(c, 0, len(d) - 1)
        dist = np.abs(t[cc] - (t + 60.0 * h))
        good = (c >= 0) & (c < len(d)) & (run[cc] == run) & (dist <= 60.0 * tol_h) & (cc != np.arange(len(d)))
        better = good & (dist < bestd)
        best[better], bestd[better] = cc[better], dist[better]
    m = best >= 0
    now, tgt = d.iloc[np.nonzero(m)[0]], d.iloc[best[m]]
    return pd.DataFrame({
        "now_idx": now.index, "tgt_idx": tgt.index, "delay_now": now.arrival_delay.to_numpy(), "delay_tgt": tgt.arrival_delay.to_numpy(),
        "t_now": now.abs_arr.to_numpy(), "t_tgt": tgt.abs_arr.to_numpy(), "train": now.train.astype(str).to_numpy(),
        "node_tgt": tgt.node.to_numpy(), "sched_hour_tgt": tgt.sched_hour.to_numpy(), "month_tgt": tgt.month.to_numpy(),
    })


def scores(y, p):
    y, p = np.asarray(y, dtype="float64"), np.asarray(p, dtype="float64")
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    err = p - y
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"n": int(len(y)), "MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "R2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")}
