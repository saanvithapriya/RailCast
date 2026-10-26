"""Item 1: primary vs secondary delay, from observable data only (SIMULATED data).

Decomposition of a section passage (run crosses one section u -> v), all in minutes. The increment runs from arrival
at the entry node u to arrival at the exit node v, so it holds the wait at u to enter the section and the running time:
    delta     = d_out - d_in                      (d_in: arrival delay at u, d_out: arrival delay at v)
    slack     = fraction(type) x scheduled section time + halt_fraction x scheduled dwell at u
    absorbed  = min(max(d_in, 0), slack)          delay a late train can recover (running slack, shortened halt at u)
    inherited = max(d_in, 0) - absorbed           carried from upstream and still there
    added     = delta + absorbed                  generated in this section (waiting for a leader, weather, incident, ...)
    d_out     = inherited + added
The scheduled slack is not in a timetable, so it is estimated from passages where it must be fully usable and nothing
else interferes (late by at least 10 minutes, no leader within 20 minutes, clear weather): the fraction of the
scheduled section time recovered (per train type) and, at stations with a scheduled halt, the fraction of the
scheduled dwell recovered.

Explanation rule and primary events:
    explained(passage) = a leader entered the same section and direction at most G minutes earlier AND the leader was
                         late (arrival delay at the end of the section >= L_late) AND the follower arrived at most G
                         minutes after the leader.
    primary event      = added >= P and not explained (a large delay that no late leading train accounts for).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Params:
    G: float = 10.0            # gap window (minutes)
    L_late: float = 5.0        # a leader is "late" if its delay at the end of the section is at least this
    A_link: float = 2.0        # a follower's added delay must be at least this to be linked to a leader
    P: float = 15.0            # primary event: added delay at least this, not explained
    T_entry: float = 15.0      # external-entry trigger: delay on entering the corridor at least this
    R_min: float = 5.0         # rake link: feeder exit delay and the run's entry delay at least this
    G_rake: float = 180.0      # rake link only if the scheduled turnaround is at most this many minutes (calibrated)
    slack_fraction: tuple = ()  # ((type, fraction), ...) estimated on the decision window and frozen

    def slack_map(self):
        return dict(self.slack_fraction)


def estimate_slack_fractions(p, min_sched_run=2.0):
    """Fraction of scheduled section time (per type) and of scheduled dwell (key _halt) that late trains recover when
    nothing else interferes. Needs attach_leaders. Returns (dict, passages used)."""
    base = p[(p.d_in >= 10) & (p.gap_act.isna() | (p.gap_act >= 20)) & (p.weather == "clear") & (p.sched_run >= min_sched_run)]
    free = base[base.dwell_u <= 0.01]                              # no scheduled halt at u: only running slack matters
    ratio = -free.delta.astype("float64") / free.sched_run.astype("float64")
    by = ratio.groupby(free.type.astype(str)).median()
    overall = float(ratio.median())
    frac = {t: float(by.get(t, overall)) for t in p.type.astype(str).unique()}
    halt = base[(base.dwell_u >= 2.0) & (base.d_in >= 20)]          # halt at u: running slack plus a shortened halt
    sec_part = halt.type.astype(str).map(frac).to_numpy() * halt.sched_run.to_numpy()
    hr = (-halt.delta.to_numpy().astype("float64") - sec_part) / halt.dwell_u.to_numpy().astype("float64")
    frac["_halt"] = float(np.median(hr)) if len(hr) else 0.0
    return frac, int(len(free) + len(halt))


def decompose(p, slack_fraction):
    """Add slack, absorbed, inherited and added columns (float32)."""
    sf = dict(slack_fraction)
    halt_frac = sf.pop("_halt", 0.0)
    out = p.copy()
    frac = out.type.astype(str).map(sf).fillna(float(np.median(list(sf.values()))))
    out["slack"] = (frac.to_numpy() * out.sched_run.to_numpy() + halt_frac * np.maximum(out.dwell_u.to_numpy(), 0)).astype("float32")
    d_in_pos = np.maximum(out.d_in.to_numpy(), 0)
    out["absorbed"] = np.minimum(d_in_pos, out.slack.to_numpy()).astype("float32")
    out["inherited"] = (d_in_pos - out.absorbed.to_numpy()).astype("float32")
    out["added"] = (out.delta.to_numpy() + out.absorbed.to_numpy()).astype("float32")
    return out


def flag_explained(p, params):
    """Boolean Series: a late leader entered shortly before and the follower arrived shortly after it. Needs attach_leaders."""
    leader_ok = p.lead_run.notna() & (p.gap_act <= params.G) & (p.lead_d_out >= params.L_late)
    close_exit = (p.t_out - p.lead_t_out) <= params.G
    return (leader_ok & close_exit).fillna(False)


def flag_primary(p, params):
    return (p.added >= params.P) & ~flag_explained(p, params)


def added_by(p, keys):
    """Mean and total added delay, and the share of passages adding at least 2 minutes, grouped by `keys`."""
    g = p.groupby(keys, observed=True)
    return pd.DataFrame({
        "passages": g.size(), "mean_added": g.added.mean(), "total_added_min": g.added.sum(),
        "mean_inherited": g.inherited.mean(), "mean_absorbed": g.absorbed.mean(),
        "share_adding_2plus": g.added.apply(lambda s: (s >= 2).mean()),
    })


GAP_EDGES = (0, 2, 4, 6, 8, 10, 15, 20, 30, 60, 120)


def gap_response_curve(p, L_late=5.0, edges=GAP_EDGES, hit_min=2.0):
    """P(follower adds >= hit_min minutes) by actual gap to the leader, for a late leader and for an on-time leader
    (the control), clear weather only. The difference is the effect of a late leader. Needs attach_leaders + decompose."""
    d = p[p.lead_run.notna() & (p.gap_act >= 0) & (p.gap_act < edges[-1]) & (p.weather == "clear")]
    d = d.assign(late=d.lead_d_out >= L_late, hit=d.added >= hit_min, b=pd.cut(d.gap_act, list(edges), right=False))
    t = d.groupby(["b", "late"], observed=True).hit.mean().unstack()
    t = t.rename(columns={True: "late_leader", False: "on_time_leader"})
    t["effect"] = t.late_leader - t.on_time_leader
    t["n_late_leader"] = d[d.late].groupby("b", observed=True).size()
    out = t.reset_index().rename(columns={"b": "gap"})
    out["gap"] = out.gap.astype(str)
    return out


def choose_gap_window(curve, edges=GAP_EDGES, frac=0.25):
    """Upper edge of the last gap bin before the first bin whose late-leader effect falls below `frac` of the first bin's
    effect. A pre-set rule, applied to the decision window only."""
    eff = curve["effect"].to_numpy()
    first = eff[0]
    for i in range(1, len(eff)):
        if eff[i] < frac * first:
            return float(edges[i])
    return float(edges[len(eff)])
