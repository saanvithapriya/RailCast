"""Item 5 (frequent propagation patterns) on hand-made cascade trees."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import patterns as pt


def tree_members():
    # cascade 0: run 0 (hit at section 5) -> run 1 (section 6) -> run 2 (section 7); run 3 hangs off run 0 at section 9
    # cascade 1: run 10 (section 5) -> run 11 (section 6) -> run 12 (section 7)
    rows = [(0, -1, 5, 0), (1, 0, 6, 0), (2, 1, 7, 0), (3, 0, 9, 0), (10, -1, 5, 1), (11, 10, 6, 1), (12, 11, 7, 1)]
    m = pd.DataFrame(rows, columns=["run_idx", "parent", "cause_sec", "cascade_id"])
    m["cause"] = np.where(m.parent >= 0, 4, 3)                       # section links below a primary root
    return m


def test_chains_pairs_and_triples():
    pair, triple = pt.chains(tree_members())
    assert sorted(zip(pair.a, pair.b)) == [(5, 6), (5, 6), (5, 9), (6, 7), (6, 7)]
    assert sorted(zip(triple.a, triple.b, triple.c)) == [(5, 6, 7), (5, 6, 7)]


def test_support_and_confidence():
    m = tree_members()
    casc = pd.DataFrame({"cascade_id": [0, 1], "depth": [2, 2]})
    p2, p3 = pt.frequent_patterns(m, casc, min_count=1)
    ab = p2.set_index(["a", "b"])
    assert ab.loc[(5, 6), "count"] == 2 and ab.loc[(5, 6), "support"] == pytest.approx(1.0)
    r = p3.set_index(["a", "b", "c"]).loc[(5, 6, 7)]
    assert r["count"] == 2 and r.support == pytest.approx(1.0) and r.confidence == pytest.approx(1.0)   # every 5->6 chain continues to 7


def test_group_specific_patterns_use_lift():
    m = tree_members()
    # add extra fog-only cascades with a pattern 20->21->22
    extra = [(100 + i * 3 + j, (100 + i * 3 + j - 1) if j else -1, 20 + j, 100 + i, 4 if j else 3) for i in range(30) for j in range(3)]
    members = pd.concat([m, pd.DataFrame(extra, columns=m.columns)], ignore_index=True)
    casc = pd.DataFrame({"cascade_id": [0, 1] + list(range(100, 130)), "depth": [2] * 32})
    groups = ["other", "other"] + ["fog"] * 30
    sup = pt.pattern_support_by_group(members, casc, groups, length=3)
    fog = pt.group_specific(sup, "fog", min_count=10)
    assert [(r.a, r.b, r.c) for r in fog.itertuples()] == [(20, 21, 22)]
    assert fog.iloc[0].support == pytest.approx(1.0)


def test_segments_and_transitions():
    anchors = [0, 10, 20, 30]
    seg = pt.segment_of_section([0, 9, 10, 25, 29], anchors)
    assert seg.tolist() == [0, 0, 1, 2, 2]
    m = tree_members().assign(cause_sec=[5, 12, 25, 5, 5, 6, 7])
    mat, stay = pt.segment_transitions(m, anchors)
    assert mat.sum() == 5 and 0.0 <= stay <= 1.0


def test_moving_versus_queue_patterns():
    m = tree_members()
    casc = pd.DataFrame({"cascade_id": [0, 1], "depth": [2, 2]})
    p2, p3 = pt.frequent_patterns(m, casc, min_count=1, moving=True)
    assert len(p3) == 1 and (p3.iloc[0].a, p3.iloc[0].b, p3.iloc[0].c) == (5, 6, 7)
    same = m.assign(cause_sec=[5, 5, 5, 5, 8, 8, 8])
    q2, q3 = pt.frequent_patterns(same, casc, min_count=1, moving=False)
    assert len(q3) == 2 and set(q2.a) == {5, 8}


def test_rake_links_are_left_out_of_propagation_patterns():
    m = tree_members()
    m.loc[m.run_idx == 3, "cause"] = 1                                # run 3 hangs off run 0 through a rake link
    pair, _ = pt.chains(m)
    assert (5, 9) not in set(zip(pair.a, pair.b))
    pair_all, _ = pt.chains(m, only_section_links=False)
    assert (5, 9) in set(zip(pair_all.a, pair_all.b))
