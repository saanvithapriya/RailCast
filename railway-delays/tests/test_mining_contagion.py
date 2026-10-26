"""Item 2 (contagion coefficients): synthetic follower/leader data with a known coefficient."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import contagion as ct


def synthetic(beta=0.5, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1e5, n))
    p = pd.DataFrame({"key": 0, "sec": 0, "dir": 1, "from_node": 0, "to_node": 1, "run_id": np.arange(n),
                      "sched_in": t, "d_in": rng.exponential(3, n), "d_ent": rng.exponential(3, n), "fast": rng.random(n) < 0.5,
                      "weather": "clear", "sched_hour": 12})
    p["type"] = "Superfast"
    lead_d = np.r_[0.0, p.d_ent.to_numpy()[:-1]]
    gap = np.r_[1e9, np.diff(t)]
    p["added"] = np.where(gap <= 10, beta * lead_d, 0.0) + rng.normal(0, 0.3, n)
    p["inherited"], p["absorbed"] = 0.0, 0.0
    p["d_out"] = p.d_in + p.added
    return p


def test_scheduled_predecessor_and_follower_counts():
    p = pd.DataFrame({"key": [0, 0, 0, 1], "sched_in": [0.0, 4.0, 30.0, 2.0], "d_in": 0.0, "d_ent": 0.0, "d_out": 0.0, "fast": True, "run_id": [1, 2, 3, 4]})
    s = ct.scheduled_neighbours(p, G=10)
    assert np.isnan(s.l_run.iloc[0]) and s.l_run.iloc[1] == 1 and s.l_run.iloc[2] == 2 and np.isnan(s.l_run.iloc[3])
    assert s.gap_sched.tolist()[1:3] == [4.0, 26.0]
    assert s.n_followers.tolist() == [1, 0, 0, 0]                  # only the 4-minute follower is inside the 10-minute window


def test_known_coefficient_is_recovered_with_a_valid_interval():
    p = synthetic(beta=0.5)
    out = ct.contagion_coefficients(p, G=10)
    r = out.iloc[0]
    assert r.n_pairs > 300
    assert r.beta_lo < 0.5 < r.beta_hi and abs(r.beta - 0.5) < 0.08
    assert r.c == pytest.approx(r.beta * r.m_followers)


def test_zero_contagion_gives_an_absorb_label_and_strong_contagion_amplifies():
    zero = ct.contagion_coefficients(synthetic(beta=0.0), G=10).iloc[0]
    assert abs(zero.beta) < 0.05 and zero.label == "absorb"
    p = synthetic(beta=2.0, n=6000)
    strong = ct.contagion_coefficients(p, G=40).iloc[0]            # a wider window gives more than one follower per leader
    assert strong.label in ("amplify", "uncertain") and strong.c > 1


def test_too_few_pairs_are_reported_without_a_coefficient():
    out = ct.contagion_coefficients(synthetic(n=200), G=10, n_min=300)
    assert np.isnan(out.iloc[0].get("beta", np.nan))


def test_pairs_are_chosen_by_the_timetable_not_by_delays():
    p = synthetic(beta=0.5)
    q = p.copy()
    q["added"] = q.added + 100                                    # changing outcomes must not change which pairs are used
    a, b = ct.contagion_coefficients(p, G=10).iloc[0], ct.contagion_coefficients(q, G=10).iloc[0]
    assert a.n_pairs == b.n_pairs and a.beta == pytest.approx(b.beta)


def test_slack_absorbed_share():
    p = pd.DataFrame({"sec": [0, 0], "dir": [1, 1], "inherited": [8.0, 0.0], "absorbed": [2.0, 0.0], "added": [0.0, 5.0]})
    t = ct.slack_absorbed(p)
    assert t.share_absorbed.iloc[0] == pytest.approx(2.0 / 15.0)


def test_pass_on_ratio_counts_followers_of_primary_events():
    from src.analysis import decomposition as dc
    prm = dc.Params(G=10, L_late=5, A_link=2, P=15)
    n = 40
    rows = []
    for i in range(n):                                              # leader passage then its follower, in the same section
        rows.append(dict(key=0, pid=2 * i, lead_pid=np.nan, d_ent=30.0, added=20.0, explained=False, primary=True))
        rows.append(dict(key=0, pid=2 * i + 1, lead_pid=2 * i, d_ent=10.0, added=10.0, explained=True, primary=False))
    p = pd.DataFrame(rows)
    out = ct.pass_on_ratio(p.drop(columns=["explained", "primary"]), p.explained, p.primary, prm, min_events=30)
    r = out.iloc[0]
    assert r.n_primary == n and r.A_primary == pytest.approx(0.5)     # 10 minutes of follower delay per 20 of leader delay
    assert r.A_primary_lo <= 0.5 <= r.A_primary_hi and r.label_pass_on == "absorb"
    assert r.rho == pytest.approx(10.0 / 30.0)                        # a third of the added minutes are explained by a leader
