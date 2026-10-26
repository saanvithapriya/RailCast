"""Phase 4 guards: decision window only, no label columns, no forbidden tables, helpers behave."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import splits
from src.analysis import eda

if not Path(splits.sources("medium")[0].dir, "observations").exists():
    pytest.skip("medium scenario not present", allow_module_level=True)


@pytest.fixture(scope="module")
def decision():
    a, b = splits.eda_windows()["decision"]
    return eda.load_window(a, b)


def test_decision_window_is_fold_1s_training_range():
    assert splits.eda_windows()["decision"] == splits.fold("fold1_fog")["train"] == ("2024-01-01", "2024-10-31")


def test_decision_data_stops_before_fold_1_validation(decision):
    df, tt, nodes, (lo, hi) = decision
    a = eda.arrivals(df, (lo, hi))
    assert a.abs_arr.min() >= lo and a.abs_arr.max() < hi
    assert a.arr_date.max() <= pd.Timestamp("2024-10-31")
    assert (a.month <= 10).all() and (a.arr_date.dt.year == 2024).all()
    assert (a.weather == "fog").sum() > 0 and a[a.arr_date >= "2024-11-01"].empty


def test_no_label_columns_are_loaded(decision):
    df = decision[0]
    for c in ("cause_code", "cause_train", "root_event_id", "primary_delay_injected"):
        assert c not in df.columns


def test_helpers_never_reference_forbidden_tables():
    """Parse the module and look at string literals and names only (docstrings and comments may mention them)."""
    import ast
    tree = ast.parse(Path("src/analysis/eda.py").read_text())
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef)) and n.body and isinstance(n.body[0], ast.Expr)
            and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
    used = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs}
    used |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for word in ("delay_attribution", "groundtruth", "events.parquet", "queues.parquet", "cause_code", "cause_train",
                 "root_event_id", "primary_delay_injected", "load_groundtruth"):
        assert not any(word in u for u in used), word
    assert "weather.parquet" in used


def test_weather_only_for_window_dates(decision):
    df = decision[0]
    assert df.weather.notna().sum() > 0
    assert df[df.arr_date > pd.Timestamp("2024-10-31")].weather.isna().all()


def test_episode_detection_and_kaplan_meier():
    x = np.array([5, 40, 35, 20, 10, 5, 50, 45, 44])
    eps = eda.disturbance_episodes(x, high=30, low=15)
    assert eps == [(1, 3, True), (6, 3, False)]          # first recovers after 3 hours (onset 1, recovery at 4); second is censored
    km = eda.kaplan_meier(np.array([3, 4]), np.array([True, False]))      # recovery at 3, censored at 4
    assert km.survival.iloc[0] == 1.0 and km.survival.iloc[-1] == pytest.approx(0.5)


def test_horizon_pairs_pick_the_right_row():
    a = pd.DataFrame({"run_id": [0, 0, 0, 1], "abs_arr": [0.0, 60.0, 130.0, 60.0], "arrival_delay": [1.0, 2.0, 3.0, 9.0],
                      "train": "T", "node": [1, 2, 3, 2], "sched_hour": 0, "month": 1})
    p = eda.horizon_pairs(a, 1, tol_h=0.25)
    pairs = set(zip(p.now_idx, p.tgt_idx))
    assert (0, 1) in pairs and (1, 2) in pairs and (3, 3) not in pairs


def test_scores_and_acf():
    s = eda.scores([1, 2, 3], [1, 2, 3])
    assert s["MAE"] == 0 and s["R2"] == 1
    assert eda.acf(np.arange(100.0), 2)[0] == pytest.approx(1.0)
