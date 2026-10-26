"""Item 1 (primary vs secondary delay): hand-made passages with known answers (SIMULATED-data methods)."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import decomposition as dc


def make_p(rows):
    """rows: dict lists with d_in, d_out, sched_run, type, weather, gap_act, lead fields (optional)."""
    p = pd.DataFrame(rows)
    p["delta"] = p.d_out - p.d_in
    for c, v in (("weather", "clear"), ("type", "Superfast")):
        if c not in p:
            p[c] = v
    if "dwell_u" not in p:
        p["dwell_u"] = 0.0
    for c in ("lead_run", "lead_d_out", "lead_t_out", "gap_act"):
        if c not in p:
            p[c] = np.nan
    if "t_out" not in p:
        p["t_out"] = 100.0
    return p


def test_decomposition_identity_and_slack_use():
    p = make_p({"d_in": [0.0, 10.0, 10.0, 0.5], "d_out": [0.0, 9.0, 40.0, 3.0], "sched_run": [10.0] * 4})
    frac = (("Superfast", 0.06),)                                   # slack 0.6 minute on a 10 minute section
    q = dc.decompose(p, frac)
    assert np.allclose(q.d_out, q.inherited + q.added)             # d_out = inherited + added, always
    assert q.absorbed.tolist() == pytest.approx([0.0, 0.6, 0.6, 0.5])  # limited by the slack and by the delay present
    assert q.added.tolist() == pytest.approx([0.0, -0.4, 30.6, 3.0])   # delta + absorbed; the last one added 2.5 on top of the 0.5 it could have recovered
    # a late train that runs free recovers exactly its slack: nothing is added
    free = dc.decompose(make_p({"d_in": [20.0], "d_out": [19.4], "sched_run": [10.0]}), frac)
    assert free.added.iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_slack_fraction_is_recovered_from_free_running_late_trains():
    n = 200
    p = make_p({"d_in": np.full(n, 30.0), "d_out": np.full(n, 30.0 - 0.08 * 10), "sched_run": np.full(n, 10.0),
                "type": ["Passenger"] * n})
    p.loc[:9, "d_out"] += 20                                        # a few passages with an extra delay do not move the median
    frac, used = dc.estimate_slack_fractions(p)
    assert frac["Passenger"] == pytest.approx(0.08, abs=1e-6) and used == n


def test_free_running_fit_ignores_contested_fog_and_on_time_passages():
    p = make_p({"d_in": [30, 30, 30, 5], "d_out": [29, 29, 29, 4], "sched_run": [10.0] * 4,
                "weather": ["clear", "fog", "clear", "clear"], "gap_act": [np.nan, np.nan, 5.0, np.nan]})
    _, used = dc.estimate_slack_fractions(p)
    assert used == 1                                                # only the first passage qualifies


def test_explained_and_primary_rules():
    prm = dc.Params(G=10, L_late=5, P=15)
    p = make_p({"d_in": [0.0] * 5, "d_out": [30.0] * 5, "sched_run": [10.0] * 5,
                "lead_run": [1, 1, 1, np.nan, 1], "lead_d_out": [20.0, 20.0, 2.0, np.nan, 20.0],
                "gap_act": [4.0, 4.0, 4.0, np.nan, 40.0], "lead_t_out": [99.0, 70.0, 99.0, np.nan, 99.0], "t_out": [100.0] * 5})
    p["added"] = [30.0] * 5
    ex = dc.flag_explained(p, prm)
    # 0: late leader, close entry, close exit -> explained; 1: leader exits 30 min earlier -> not explained;
    # 2: leader not late -> not explained; 3: no leader -> not explained; 4: leader entered 40 min earlier -> not explained
    assert ex.tolist() == [True, False, False, False, False]
    assert dc.flag_primary(p, prm).tolist() == [False, True, True, True, True]
    small = p.assign(added=[5.0] * 5)
    assert not dc.flag_primary(small, prm).any()                   # below the primary threshold


def test_gap_window_rule():
    curve = pd.DataFrame({"effect": [0.04, 0.03, 0.02, 0.012, 0.008, 0.002, 0.0, 0.0, 0.0, 0.0]})
    # first bin effect 0.04; 25% = 0.01; the first bin below that is index 4 (0.008), whose lower edge is 8
    assert dc.choose_gap_window(curve) == 8.0


def test_added_by_groups():
    p = dc.decompose(make_p({"d_in": [0.0, 0.0, 0.0, 0.0], "d_out": [5.0, 1.0, 10.0, 0.0], "sched_run": [10.0] * 4,
                             "type": ["Passenger", "Passenger", "Superfast", "Superfast"]}), (("Passenger", 0.0), ("Superfast", 0.0)))
    t = dc.added_by(p, ["type"])
    assert t.loc["Passenger", "mean_added"] == pytest.approx(3.0) and t.loc["Superfast", "total_added_min"] == pytest.approx(10.0)


def test_halt_slack_and_entry_wait_are_part_of_the_section_increment():
    # a late train at a station with a scheduled 10-minute dwell recovers half of it (5 minutes) plus 6% of the section
    frac = (("Superfast", 0.06), ("_halt", 0.5))
    p = make_p({"d_in": [30.0, 30.0], "d_out": [30.0 - 5.0 - 0.6, 30.0 + 7.0 - 5.0 - 0.6], "sched_run": [10.0, 10.0], "dwell_u": [10.0, 10.0]})
    q = dc.decompose(p, frac)
    assert q.slack.tolist() == pytest.approx([5.6, 5.6])
    assert q.added.tolist() == pytest.approx([0.0, 7.0], abs=1e-5)           # the second train waited 7 minutes at the entry node
    assert np.allclose(q.d_out, q.inherited + q.added)


def test_halt_fraction_is_estimated_from_late_trains_at_halting_stations():
    n = 100
    p = make_p({"d_in": np.full(n, 40.0), "d_out": np.full(n, 40.0 - 0.06 * 10 - 0.5 * 8), "sched_run": np.full(n, 10.0),
                "dwell_u": np.r_[np.full(n // 2, 8.0), np.zeros(n // 2)]})
    p.loc[n // 2:, "d_out"] = 40.0 - 0.6                            # no halt: running slack only
    p["delta"] = p.d_out - p.d_in
    frac, _ = dc.estimate_slack_fractions(p)
    assert frac["Superfast"] == pytest.approx(0.06, abs=1e-6) and frac["_halt"] == pytest.approx(0.5, abs=1e-6)
