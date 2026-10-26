"""PROPOSED v2 of the primary-event rule (SIMULATED data). v1 (decomposition.flag_primary) stays frozen and unchanged.

Why v2: the validation (notebook 07) showed that 38% of the primary events v1 reports are really trains held for a faster
train (yields). A held train enters the section AFTER the faster train, so in the data the faster train is its leader and it
is on time; v1 only accepts a LATE leader as an explanation. v2 adds one observable rule:

    yield-explained(passage) = the previous train to enter the same section and direction entered at most G_lead minutes
                               earlier and is of a higher class (Rajdhani > Duronto/Shatabdi > Superfast > Mail/Express >
                               Passenger) than this train
    primary_v2 = added >= P  and  not explained by a late leader (v1)  and  not yield-explained

G_lead is calibrated on the decision window from observable data only (the gap where a faster on-time leader still raises
the chance of a large delay behind it), like G. v2 was designed after seeing the validation results on 2025-01-01..2025-08-31,
so its numbers on that window are optimistic; it is checked on 2024-11-01..2024-12-31, which the design did not use. It is a
proposal, not a frozen method.
"""
from pathlib import Path

import pandas as pd
import yaml

from . import decomposition as dc

CLASS_RANK = {"Rajdhani": 1, "Duronto": 2, "Shatabdi": 2, "Superfast": 3, "Mail/Express": 4, "Passenger": 5, "other": 4}
LEAD_EDGES = (0, 2, 4, 6, 8, 10, 15, 20, 30, 60)
V2_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "mining_v2.yaml"


def _leader_class(p):
    """Class rank of every passage and of its leader (previous entry, same section and direction)."""
    rank = p.type.astype(str).map(CLASS_RANK)
    lead_rank = rank.groupby(p.key.to_numpy()).shift(1)
    return rank.to_numpy(), lead_rank.to_numpy()


def leader_curve(p, hit_min=5.0, edges=LEAD_EDGES):
    """P(added >= hit_min) by the actual gap to the leader, when the leader is of a higher class and ON TIME (not late) and
    when it is not of a higher class (the control), clear weather. The difference is the effect of a faster on-time leader."""
    rank, lead_rank = _leader_class(p)
    d = pd.DataFrame({"gap": p.gap_act.to_numpy(), "faster": lead_rank < rank, "hit": (p.added >= hit_min).to_numpy(),
                      "on_time": (p.lead_d_out < 5.0).to_numpy(), "clear": (p.weather == "clear").to_numpy()})
    d = d[d.clear & d.on_time & (d.gap >= 0) & (d.gap < edges[-1])]
    d["b"] = pd.cut(d.gap, list(edges), right=False)
    t = d.groupby(["b", "faster"], observed=True).hit.mean().unstack().rename(columns={True: "faster_leader", False: "not_faster_leader"})
    t["effect"] = t.faster_leader - t.not_faster_leader
    t["n_faster"] = d[d.faster].groupby("b", observed=True).size()
    out = t.reset_index().rename(columns={"b": "gap"})
    out["gap"] = out.gap.astype(str)
    return out


def choose_leader_window(curve, edges=LEAD_EDGES, frac=0.25):
    """Upper edge of the last bin before the first bin whose effect falls below `frac` of the largest bin effect
    (the same pre-set style of rule as the leader gap window)."""
    eff = curve["effect"].to_numpy()
    peak = int(eff.argmax())
    for i in range(peak + 1, len(eff)):
        if eff[i] < frac * eff[peak]:
            return float(edges[i])
    return float(edges[len(eff)])


def load_leader_window():
    return float(yaml.safe_load(V2_CONFIG.read_text())["G_lead"])


def flag_yield_explained(p, g_lead):
    """True where the leader entered at most g_lead minutes earlier and is of a higher class. Needs attach_leaders."""
    rank, lead_rank = _leader_class(p)
    return pd.Series((p.gap_act.to_numpy() <= g_lead) & (lead_rank < rank), index=p.index).fillna(False)


def primary_v1_v2(p, params, g_lead=None):
    """(primary_v1, primary_v2) boolean Series. g_lead defaults to the value calibrated on the decision window."""
    g_lead = load_leader_window() if g_lead is None else g_lead
    explained = dc.flag_explained(p, params)
    v1 = (p.added >= params.P) & ~explained
    v2 = v1 & ~flag_yield_explained(p, g_lead)
    return v1, v2


def calibrate_v2():
    """Choose G_trail on the DECISION window only and write configs/mining_v2.yaml (once)."""
    from src import splits
    from . import eda, passages
    from .calibrate import load_frozen

    a, b = splits.eda_windows()["decision"]
    prm = load_frozen()
    df, tt, nodes, bounds = eda.load_window(a, b)
    p = dc.decompose(passages.attach_leaders(passages.build_passages(df, bounds)), prm.slack_fraction)
    curve = leader_curve(p)
    g = choose_leader_window(curve)
    V2_CONFIG.write_text("# PROPOSED v2 parameter, calibrated on 2024-01-01..2024-10-31 only (observable data, no ground truth).\n"
                         + yaml.safe_dump({"G_lead": g, "window": [a, b], "curve": curve.round(4).to_dict("records")}, sort_keys=False))
    return g, curve


if __name__ == "__main__":
    g, c = calibrate_v2()
    print("G_lead =", g)
    print(c.round(3).to_string(index=False))
