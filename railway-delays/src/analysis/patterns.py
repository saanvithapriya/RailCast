"""Item 5: frequent propagation patterns in the extracted cascades (SIMULATED data, observable inputs only).

Every cascade member run has a "hit section" (cause_sec): where it was delayed by its parent (section link), where its
rake service started (rake link), or where the trigger happened (root). Following a chain of parents gives a sequence of
hit sections root -> ... -> run. A pattern is a consecutive subsequence A->B (length 2) or A->B->C (length 3) of such a
sequence. For each pattern:
    count       number of parent-child chains showing it
    support     share of cascades (deep enough to contain it) that show it at least once
    confidence  for A->B->C: count(A->B->C) / count(A->B)   (how often a chain through A->B continues to C)
Patterns can be computed per season group and compared: lift = support in the group / support elsewhere.
"""
import numpy as np
import pandas as pd


def chains(members, only_section_links=True):
    """Parent-child pairs (A, B) and parent-grandparent-child triples (A, B, C) of hit sections, with cascade id.

    only_section_links=True keeps edges where the child was delayed by a late leader in a section (cause 4), so the
    patterns describe propagation between trains on the line; rake links (feeder to next service, cause 1) connect the
    end of one journey to the start of the next and are left out."""
    m = members.set_index("run_idx")
    e = m[m.parent >= 0]
    if only_section_links and "cause" in m:
        e = e[e.cause == 4]
    ok = e.parent.isin(m.index)
    e = e[ok]
    pair = pd.DataFrame({"cascade_id": e.cascade_id.to_numpy(), "a": m.cause_sec.reindex(e.parent).to_numpy(), "b": e.cause_sec.to_numpy(),
                         "child": e.index.to_numpy(), "parent": e.parent.to_numpy()})
    gp = m.parent.reindex(e.parent).to_numpy()
    if only_section_links and "cause" in m:
        gp = np.where(m.cause.reindex(e.parent).to_numpy() == 4, gp, np.nan)      # the parent must itself be a section link
    has = ~np.isnan(gp) & (gp >= 0)
    t = pair[has].copy()
    t["a0"] = m.cause_sec.reindex(gp[has].astype(int)).to_numpy()
    triple = pd.DataFrame({"cascade_id": t.cascade_id.to_numpy(), "a": t.a0.to_numpy(), "b": t.a.to_numpy(), "c": t.b.to_numpy()})
    return pair, triple


def frequent_patterns(members, cascades, min_count=20, moving=None, only_section_links=True):
    """Pattern tables for length 2 and 3 over all cascades: count, support, and (length 3) confidence.

    moving=None keeps every pattern; moving=True keeps only patterns that change section at every step (A->B, A->B->C with
    all sections different between neighbours: propagation along the line); moving=False keeps only same-section
    repeats (a queue of trains behind a late leader).
    """
    pair, triple = chains(members, only_section_links)
    if moving is not None:
        pair = pair[(pair.a != pair.b) == moving]
        triple = triple[((triple.a != triple.b) & (triple.b != triple.c)) == moving] if moving else triple[(triple.a == triple.b) & (triple.b == triple.c)]
    n2 = int((cascades.depth >= 1).sum())
    n3 = int((cascades.depth >= 2).sum())
    p2 = pair.groupby(["a", "b"]).agg(count=("cascade_id", "size"), cascades=("cascade_id", "nunique")).reset_index()
    p2["support"] = p2.cascades / max(n2, 1)
    p3 = triple.groupby(["a", "b", "c"]).agg(count=("cascade_id", "size"), cascades=("cascade_id", "nunique")).reset_index()
    p3["support"] = p3.cascades / max(n3, 1)
    p3 = p3.merge(p2[["a", "b", "count"]].rename(columns={"a": "a", "b": "b", "count": "count_ab"}), on=["a", "b"], how="left")
    p3["confidence"] = p3["count"] / p3.count_ab
    return p2[p2["count"] >= min_count].sort_values("count", ascending=False), p3[p3["count"] >= min_count].sort_values("count", ascending=False)


def pattern_support_by_group(members, cascades, group_of_cascade, length=3, only_section_links=True):
    """Support of every length-2/3 pattern within each group of cascades (e.g. season). Returns a long DataFrame."""
    pair, triple = chains(members, only_section_links)
    grp = pd.Series(group_of_cascade, index=cascades.cascade_id.to_numpy())
    need = 1 if length == 2 else 2
    base = cascades.assign(g=grp.reindex(cascades.cascade_id).to_numpy())
    base = base[base.depth >= need].groupby("g").size()
    t = (pair if length == 2 else triple)
    t = t.assign(g=grp.reindex(t.cascade_id).to_numpy())
    keys = ["a", "b"] if length == 2 else ["a", "b", "c"]
    c = t.groupby(keys + ["g"]).agg(count=("cascade_id", "size"), cascades=("cascade_id", "nunique")).reset_index()
    c["support"] = c.cascades / c.g.map(base)
    return c


def group_specific(support_long, group, min_count=20, min_lift=2.0):
    """Patterns far more common in `group` than elsewhere: lift = support in the group / support in all other groups."""
    keys = [c for c in support_long.columns if c in ("a", "b", "c")]
    inn = support_long[support_long.g == group].set_index(keys)
    out = support_long[support_long.g != group].groupby(keys).agg(count_other=("count", "sum"), cascades_other=("cascades", "sum"))
    j = inn.join(out, how="left").fillna({"count_other": 0, "cascades_other": 0})
    # support elsewhere is pooled over the other groups using their cascade totals
    tot = support_long.groupby("g").apply(lambda d: (d.cascades / d.support).max(), include_groups=False)
    other_total = tot[tot.index != group].sum()
    j["support_other"] = j.cascades_other / other_total
    j["lift"] = j.support / j.support_other.replace(0, np.nan)
    j = j[(j["count"] >= min_count) & ((j.lift >= min_lift) | j.lift.isna())]
    return j.sort_values("support", ascending=False).reset_index()


def segment_of_section(sections, anchor_nodes):
    """Physical segment index of each section given the chain indices of the anchors (NDLS, TDL, ETW, CNB, PRYG, MGS)."""
    edges = list(anchor_nodes)
    return np.clip(np.searchsorted(edges, np.asarray(sections), side="right") - 1, 0, len(edges) - 2)


def segment_transitions(members, anchor_nodes, only_section_links=True):
    """How often a chain moves from segment i to segment j (counts), with the share staying inside the segment."""
    pair, _ = chains(members, only_section_links)
    sa, sb = segment_of_section(pair.a, anchor_nodes), segment_of_section(pair.b, anchor_nodes)
    k = len(anchor_nodes) - 1
    mat = np.zeros((k, k), dtype=int)
    np.add.at(mat, (sa, sb), 1)
    return mat, float(np.trace(mat) / max(mat.sum(), 1))
