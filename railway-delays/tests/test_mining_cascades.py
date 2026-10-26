"""Item 3 (cascade extraction): a hand-made scenario with known trees."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import cascades as cs
from src.analysis import decomposition as dc

PRM = dc.Params(G=10, L_late=5, A_link=2, P=15, T_entry=15, R_min=5)


def passage(run, t, sec, added, lead=np.nan, weather="clear"):
    return dict(run_id=run, t_in=t, t_out=t + 5, sec=sec, added=added, lead_run=lead, weather=weather)


def runs_table(rows):
    r = pd.DataFrame(rows).set_index("run_id")
    for c, v in (("feeder_train", None), ("feeder_k", 0), ("entry_delay", 0.0), ("exit_delay", 0.0), ("from_start", True), ("entry_time", 0.0), ("day", 0)):
        if c not in r:
            r[c] = v
    r["train"] = [f"T{i}" for i in r.index]
    return r


def test_tree_with_primary_root_children_and_a_grandchild():
    # run 1: primary event (20 min, unexplained); run 2 follows run 1 (adds 10); run 3 follows run 2 (adds 5);
    # run 4 adds only 1 minute behind run 1 (below A_link); run 5 is unrelated and small
    p = pd.DataFrame([passage(1, 100, 7, 20), passage(2, 110, 7, 10, lead=1), passage(3, 120, 8, 5, lead=2),
                      passage(4, 112, 7, 1, lead=1), passage(5, 300, 9, 3)])
    p["explained"] = [False, True, True, True, False]
    p["primary"] = [True, False, False, False, False]
    runs = runs_table([dict(run_id=i) for i in (1, 2, 3, 4, 5)])
    c, m, links = cs.extract_cascades(p, runs, PRM, p.explained, p.primary)
    assert len(c) == 1
    r = c.iloc[0]
    assert r.size_runs == 3 and r.depth == 2 and r.origin_type == "primary" and r.origin_sec == 7 and r.origin_time == 100
    assert r.extent_sections == 2 and r.total_delay_min == pytest.approx(20 + 10 + 5) and r.duration_min == pytest.approx(125 - 100)
    assert sorted(m.run_id) == [1, 2, 3] and 4 not in m.run_id.tolist() and 5 not in m.run_id.tolist()
    assert sorted(links.child_run_id) == [2, 3] and sorted(links.parent_run_id) == [1, 2]


def test_a_chain_that_ends_in_an_unexplained_late_leader_is_not_a_cascade():
    p = pd.DataFrame([passage(1, 100, 7, 3), passage(2, 110, 7, 10, lead=1)])        # run 1 late but no trigger anywhere
    p["explained"], p["primary"] = [False, True], [False, False]
    c, m, _ = cs.extract_cascades(p, runs_table([dict(run_id=1), dict(run_id=2)]), PRM, p.explained, p.primary)
    assert len(c) == 0 and len(m) == 0


def test_rake_link_and_external_entry():
    p = pd.DataFrame([passage(1, 100, 7, 20), passage(2, 400, 5, 0.0)])
    p["explained"], p["primary"] = [False, False], [True, False]
    runs = runs_table([
        dict(run_id=1, entry_time=90.0, exit_delay=25.0),                                                    # root through its primary event
        dict(run_id=2, entry_time=380.0, entry_delay=12.0, feeder_train="T1", feeder_k=0),                   # late by 12, feeder left 25 late
        dict(run_id=3, entry_time=500.0, entry_delay=30.0),                                                  # enters 30 late, no feeder: external root
        dict(run_id=4, entry_time=600.0, entry_delay=12.0, feeder_train="T3", feeder_k=0, exit_delay=0.0),   # feeder run 3 left on time here
    ])
    c, m, _ = cs.extract_cascades(p, runs, PRM, p.explained, p.primary)
    kinds = c.set_index("origin_run_id").origin_type.to_dict()
    assert kinds == {1: "primary", 3: "external_entry"}
    big = c.set_index("origin_run_id").loc[1]
    assert big.size_runs == 2 and big.depth == 1                     # run 2 hangs below run 1 through the rake link
    assert 4 not in m.run_id.tolist()                                # feeder was not late enough to explain it


def test_earliest_cause_wins_and_ties_prefer_the_entry_causes():
    p = pd.DataFrame([passage(1, 100, 7, 20), passage(2, 200, 7, 20)])
    p["explained"], p["primary"] = [False, False], [True, True]
    # run 2 has a primary event at t=200 but entered 40 late at t=50: the external entry is the earlier cause
    runs = runs_table([dict(run_id=1), dict(run_id=2, entry_time=50.0, entry_delay=40.0)])
    c, m, _ = cs.extract_cascades(p, runs, PRM, p.explained, p.primary)
    assert m.set_index("run_id").loc[2, "cause_name"] == "external_entry"


def test_summary_and_sensitivity_run():
    p = pd.DataFrame([passage(1, 100, 7, 20), passage(2, 110, 7, 10, lead=1)])
    p["explained"], p["primary"] = [False, True], [True, False]
    p["lead_d_out"], p["lead_t_out"], p["gap_act"], p["d_ent"] = np.nan, np.nan, np.nan, 0.0
    c, _, _ = cs.extract_cascades(p, runs_table([dict(run_id=1), dict(run_id=2)]), PRM, p.explained, p.primary)
    s = cs.summarise(c)
    assert s["cascades"] == 1 and s["max_size"] == 2 and s["share_size_2plus"] == 1.0


def test_rake_window_is_chosen_where_the_feeder_relationship_ends():
    from src.analysis.calibrate import choose_rake_window
    rng = np.random.default_rng(1)
    rows = []
    for g, linked in ((60, True), (100, True), (150, True), (200, False), (300, False), (500, False)):
        for i in range(800):
            exitd = rng.exponential(20)
            entry = exitd + rng.normal(0, 2) if linked else rng.exponential(20)
            rows.append(dict(train=f"F{g}_{i}", day=0, exit_delay=exitd, entry_delay=0.0, feeder_train=None, feeder_k=0, feeder_gap=np.nan, from_start=True))
            rows.append(dict(train=f"R{g}_{i}", day=0, exit_delay=0.0, entry_delay=entry, feeder_train=f"F{g}_{i}", feeder_k=0, feeder_gap=float(g), from_start=True))
    runs = pd.DataFrame(rows)
    G, curve = choose_rake_window(runs, None)
    assert G == 180.0                                                  # the bin (180, 240] is the first judged bin below 0.3
    assert curve[0]["pairs"] == 800 and curve[0]["corr_entry_vs_feeder_exit"] > 0.8            # the bin (0, 60]
    assert all(c["corr_entry_vs_feeder_exit"] < 0.3 for c in curve if c["gap_from"] >= 180 and c["pairs"] > 0)


def test_a_long_turnaround_does_not_make_a_rake_link():
    p = pd.DataFrame([passage(1, 100, 7, 20)])
    p["explained"], p["primary"] = [False], [True]
    runs = runs_table([dict(run_id=1, entry_time=90.0, exit_delay=25.0),
                       dict(run_id=2, entry_time=500.0, entry_delay=12.0, feeder_train="T1", feeder_k=0, feeder_gap=400.0)])
    c, m, _ = cs.extract_cascades(p, runs, PRM, p.explained, p.primary)
    assert 2 not in m.run_id.tolist() and len(c) == 1 and c.iloc[0].size_runs == 1
