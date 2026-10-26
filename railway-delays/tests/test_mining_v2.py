"""The proposed v2 yield rule on a tiny hand-made section."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import decomposition as dc
from src.analysis import mining_v2 as v2

PRM = dc.Params(G=6, L_late=5, P=15)


def table(rows):
    """rows: (key, t_in, type, added, gap_act, lead_d_out), in entry order within a key."""
    p = pd.DataFrame(rows, columns=["key", "t_in", "type", "added", "gap_act", "lead_d_out"])
    p["lead_run"] = np.where(p.gap_act.notna(), 1.0, np.nan)
    p["lead_t_out"], p["t_out"] = p.t_in - p.gap_act + 5, p.t_in + 5
    p["weather"] = "clear"
    return p


def test_a_delayed_train_right_behind_an_on_time_faster_one_is_yield_explained():
    p = table([(0, 100.0, "Rajdhani", 0.0, np.nan, np.nan), (0, 106.0, "Passenger", 25.0, 6.0, 0.0),        # held for the Rajdhani: explained
               (1, 300.0, "Passenger", 0.0, np.nan, np.nan), (1, 306.0, "Passenger", 25.0, 6.0, 0.0),       # same class ahead: not a yield
               (2, 500.0, "Passenger", 0.0, np.nan, np.nan), (2, 506.0, "Rajdhani", 25.0, 6.0, 0.0),        # slower train ahead: not a yield
               (3, 700.0, "Rajdhani", 0.0, np.nan, np.nan), (3, 730.0, "Passenger", 25.0, 30.0, 0.0)])      # the faster train is 30 minutes ahead
    y = v2.flag_yield_explained(p, 8)
    assert y.tolist() == [False, True, False, False, False, False, False, False]
    v1, vv2 = v2.primary_v1_v2(p, PRM, g_lead=8)
    assert v1.sum() == 4 and vv2.sum() == 3                                       # only the first large delay is explained away


def test_v1_is_untouched():
    import inspect
    assert "yield" not in inspect.getsource(dc.flag_primary)                      # the frozen rule knows nothing about v2


def test_leader_window_rule():
    curve = pd.DataFrame({"effect": [0.02, 0.10, 0.12, 0.09, 0.02, 0.01, 0.0, 0.0, 0.0]})
    assert v2.choose_leader_window(curve) == 8.0                                    # peak at bin 2; bin 4 (0.02) is below 25% of 0.12 -> edge 8


def test_leader_curve_shows_the_effect_of_a_faster_on_time_leader():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(4000):
        gap = rng.uniform(0, 30)
        faster = rng.random() < 0.5
        added = 20.0 if (faster and 4 <= gap < 8 and rng.random() < 0.6) else 0.0
        rows.append((i, 1000.0 * i, "Rajdhani" if faster else "Passenger", 0.0, np.nan, np.nan))
        rows.append((i, 1000.0 * i + gap, "Passenger", added, gap, 0.0))
    p = table(rows)
    c = v2.leader_curve(p)
    assert c.effect.max() > 0.3 and abs(c.effect.iloc[-1]) < 0.05
    assert v2.choose_leader_window(c) in (8.0, 10.0)
