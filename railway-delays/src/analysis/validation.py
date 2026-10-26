"""Validation of the mined cascades against the simulator's ground truth (SIMULATED data, VALIDATION ONLY).

*** Nothing in the mining pipeline imports this module. It reads delay_attribution and the ground-truth cascades, which a
real dataset would not have, and only the separate validation notebook uses it. Every loader refuses unless it is called
with validation_only=True. ***

Truth links come from the simulator's cause labels at the arrival row of the downstream station of a section:
  cause 4 section wait (a leading train held the follower), 5 yield (the train was held for a faster train that arrives
  after it), 7 rake reuse, 1 incident, 2 disruption, 3 weather, 6 freight, 8 halt extension, 9 entry delay.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.graph import delay_attrs

OTHER_TRAIN = (4, 5)
GENERATED = (1, 2, 3, 6, 8)             # delay generated in the section (not caused by another train, not at entry)
NAMES = {1: "incident", 2: "disruption", 3: "weather", 4: "section wait (leader)", 5: "yield", 6: "freight", 7: "rake reuse", 8: "halt extension", 9: "entry delay"}


def _guard(validation_only):
    if not validation_only:
        raise PermissionError("validation code reads the ground truth; pass validation_only=True from the validation notebook only")


def load_truth_links(scenario_dir, start, end, node_of, directions, min_minutes=0.5, *, validation_only=False):
    """One row per (follower arrival row, cause) with at least `min_minutes` added: train, day, to_node, from_node, sec,
    dir, cause_code, cause_train, new_minutes, event_id, root_event_id, t (arrival time, absolute minutes)."""
    _guard(validation_only)
    scen = Path(scenario_dir)
    lo, hi = delay_attrs.window_minutes(start, end)
    months = sorted(p.name.split("=")[1] for p in (scen / "observations").glob("month=*"))
    parts = []
    for m in months:
        obs = pd.read_parquet(scen / "observations" / f"month={m}" / "part.parquet", columns=["obs_id", "train", "date", "station", "actual_arrival"])
        at = pd.read_parquet(scen / "delay_attribution" / f"month={m}" / "part.parquet")
        at = at[(at.phase == "arr") & (at.cause_code > 0) & (at.cause_code < 10) & (at.new_minutes >= min_minutes)]
        j = at.merge(obs, on="obs_id")
        day = (pd.to_datetime(j.date) - pd.Timestamp(delay_attrs.EPOCH)).dt.days
        j = j.assign(day=day.to_numpy(), t=day.to_numpy() * 1440.0 + j.actual_arrival.astype("float64"))
        parts.append(j[(j.t >= lo) & (j.t < hi)])
    d = pd.concat(parts, ignore_index=True)
    d["train"], d["cause_train"] = d.train.astype(str), d.cause_train.astype(str).replace("nan", "")
    d["to_node"] = d.station.astype(str).map(node_of)
    d["dir"] = d.train.map(directions)
    d["from_node"] = d.to_node - d["dir"]
    d["sec"] = np.minimum(d.to_node, d.from_node)
    return d[["train", "day", "to_node", "from_node", "sec", "dir", "cause_code", "cause_train", "new_minutes", "event_id", "root_event_id", "t"]]


def link_precision_recall(inferred, truth, codes=OTHER_TRAIN):
    """Inferred links (columns train, day, to_node, leader) against true cause links with a cause code in `codes`.

    A link is correct if the follower run, the downstream node and the leader train all match a true link."""
    tr = truth[truth.cause_code.isin(codes)][["train", "day", "to_node", "cause_train"]].drop_duplicates().rename(columns={"cause_train": "leader"})
    inf = inferred[["train", "day", "to_node", "leader"]].drop_duplicates()
    hit = inf.merge(tr, on=["train", "day", "to_node", "leader"], how="inner")
    return {"inferred": len(inf), "true": len(tr), "correct": len(hit), "precision": len(hit) / max(len(inf), 1), "recall": len(hit) / max(len(tr), 1)}


def recall_by_cause(inferred, truth):
    """Share of true links of each cause code that the inferred links recover (follower, node, leader train)."""
    rows = []
    inf = inferred[["train", "day", "to_node", "leader"]].drop_duplicates()
    for code, g in truth.groupby("cause_code"):
        g = g[["train", "day", "to_node", "cause_train"]].drop_duplicates().rename(columns={"cause_train": "leader"})
        hit = g.merge(inf, on=["train", "day", "to_node", "leader"], how="inner")
        rows.append({"cause_code": int(code), "cause": NAMES.get(int(code), str(code)), "true_links": len(g), "recovered": len(hit), "recall": len(hit) / max(len(g), 1)})
    return pd.DataFrame(rows)


def false_positive_breakdown(inferred, truth):
    """For inferred links that are not correct: was there a true other-train cause with a different train (wrong leader),
    a true cause that is not another train, or no true cause at that row?"""
    inf = inferred[["train", "day", "to_node", "leader"]].drop_duplicates()
    ok = truth[truth.cause_code.isin(OTHER_TRAIN)][["train", "day", "to_node", "cause_train"]].drop_duplicates().rename(columns={"cause_train": "leader"})
    ok["correct"] = True
    j = inf.merge(ok, on=["train", "day", "to_node", "leader"], how="left")
    wrong = j[j.correct.isna()].drop(columns="correct")
    key = ["train", "day", "to_node"]
    any_other = truth[truth.cause_code.isin(OTHER_TRAIN)][key].drop_duplicates().assign(other=True)
    any_cause = truth[key].drop_duplicates().assign(cause=True)
    w = wrong.merge(any_other, on=key, how="left").merge(any_cause, on=key, how="left")
    cat = np.where(w.other.notna(), "wrong leader (another train was the cause)", np.where(w.cause.notna(), "caused by something other than a train", "no label at that row"))
    return pd.Series(cat).value_counts()


def origin_cause_breakdown(primary, truth):
    """What really caused the inferred primary events: the cause with most added minutes at that run, section."""
    key = ["train", "day", "sec"]
    t = truth.sort_values("new_minutes", ascending=False).drop_duplicates(key)[key + ["cause_code"]]
    j = primary.merge(t, on=key, how="left")
    j["cause"] = j.cause_code.map(NAMES).fillna("no label (below 0.5 min)")
    return j.cause.value_counts(), j


def match_origins(casc, events, tol=5.0):
    """Match inferred primary-origin cascades to true section events by train, section and start time (minutes).

    casc: origin_train, origin_sec, origin_time; events: train, section, start_min, event_id."""
    c = casc.reset_index(drop=True).assign(_i=lambda d: d.index)
    j = c.merge(events[["event_id", "train", "section", "start_min"]], left_on=["origin_train", "origin_sec"], right_on=["train", "section"], how="inner")
    j = j[(j.origin_time - j.start_min).abs() <= tol]
    j["dt"] = (j.origin_time - j.start_min).abs()
    return j.sort_values("dt").drop_duplicates("_i")


def spearman(a, b):
    a, b = pd.Series(a), pd.Series(b)
    return float(a.corr(b, method="spearman"))


def other_train_share(truth, by=("sec", "dir")):
    """True share of the delay generated on a section (causes 1-6, 8) that comes from other trains (causes 4 and 5)."""
    g = truth[truth.cause_code.isin(GENERATED + OTHER_TRAIN)]
    tot = g.groupby(list(by)).new_minutes.sum()
    oth = g[g.cause_code.isin(OTHER_TRAIN)].groupby(list(by)).new_minutes.sum().reindex(tot.index).fillna(0)
    return (oth / tot).rename("true_other_share").reset_index()
