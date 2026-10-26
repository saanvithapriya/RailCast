"""Strict leakage prevention, temporal isolation, and anti-leakage verification tests."""
import copy
import numpy as np
import pandas as pd
import pytest

from src import splits
from src.models.dataset import compute_downstream_targets
from src.models.features import FEATURE_COLUMNS, FeaturePipeline, extract_observation_features


def test_splits_match_config_and_do_not_overlap():
    """Verify that prediction splits strictly adhere to configs/splits.yaml."""
    tr_start, tr_end = splits.window("train")
    va_start, va_end = splits.window("validation")
    te_start, te_end = splits.window("test")

    assert tr_start == "2024-01-01" and tr_end == "2025-08-31"
    assert va_start == "2025-09-01" and va_end == "2025-10-31"
    assert te_start == "2025-11-01" and te_end == "2025-12-31"

    # Strict contiguous chronological boundaries without overlap
    d = pd.Timestamp
    assert d(tr_end) + pd.Timedelta(days=1) == d(va_start)
    assert d(va_end) + pd.Timedelta(days=1) == d(te_start)


def test_hidden_simulator_labels_are_not_in_features():
    """Verify that forbidden internal simulator columns never appear in FEATURE_COLUMNS."""
    forbidden = {
        "cause_code",
        "cause_train",
        "root_event_id",
        "primary_delay_injected",
        "groundtruth_cascades",
        "delay_attribution",
        "inferred_cascade_id",
    }
    for col in FEATURE_COLUMNS:
        assert col not in forbidden, f"Forbidden column '{col}' found in feature list!"


def test_preprocessor_is_independent_of_test_and_validation_data():
    """Modifying future or validation data must have ZERO effect on learned training parameters."""
    train_rows = [
        {"run_id": 1, "current_node": 10, "current_delay": 5.0, "seq": 1, "abs_arr": 100.0, "type": "Superfast", "priority": 3, "weather": "clear", "train": "100"},
        {"run_id": 1, "current_node": 10, "current_delay": 15.0, "seq": 2, "abs_arr": 200.0, "type": "Superfast", "priority": 3, "weather": "clear", "train": "100"},
    ]
    train_df = pd.DataFrame(train_rows)

    pipeline1 = FeaturePipeline().fit(train_df)

    # Now create a "tampered" future dataframe with extreme values (e.g. 9999 delay)
    val_rows = [
        {"run_id": 2, "current_node": 10, "current_delay": 9999.0, "seq": 1, "abs_arr": 3000.0, "type": "Superfast", "priority": 3, "weather": "fog", "train": "100"},
    ]
    val_df = pd.DataFrame(val_rows)

    pipeline2 = FeaturePipeline().fit(train_df)

    # Preprocessor 1 and 2 must be identical because neither saw val_df during fit
    assert pipeline1.node_hist_mean[10] == pipeline2.node_hist_mean[10] == pytest.approx(10.0)
    assert pipeline1.overall_mean_delay == pipeline2.overall_mean_delay == pytest.approx(10.0)

    # When transforming val_df, pipeline must use the TRAINING learned mean (10.0), NOT 9999.0
    X_val = pipeline1.transform(val_df)
    node_hist_idx = FEATURE_COLUMNS.index("node_hist_mean_delay")
    assert X_val[0, node_hist_idx] == pytest.approx(10.0)


def test_intentional_future_leakage_in_target_detection():
    """Verify that targets never use past or contemporaneous events (must be strictly in the future)."""
    rows = [
        {"run_id": 1, "seq": 0, "abs_arr": 100.0, "arrival_delay": 5.0, "train": "1"},
        {"run_id": 1, "seq": 1, "abs_arr": 160.0, "arrival_delay": 15.0, "train": "1"},
    ]
    df = pd.DataFrame(rows)
    df_tgt = compute_downstream_targets(df, horizons=[1], tol_min=35.0)

    # For stop 0 (t=100), target at 1h (t=160) is 15.0
    assert df_tgt.loc[0, "target_delay_1h"] == pytest.approx(15.0)

    # For stop 1 (t=160), target at 1h must NOT leak stop 0 or stop 1; it has no future event -> must be NaN
    assert np.isnan(df_tgt.loc[1, "target_delay_1h"])


def test_station_lead_delay_cannot_use_future_arrivals():
    """Observable preceding train delay at a station must have arrived BEFORE t, never after."""
    # Train A arrives at station node 5 at t=100 with delay=10
    # Train B arrives at station node 5 at t=120 with delay=50
    # Train C arrives at station node 5 at t=150 with delay=70
    rows = [
        {"run_id": 1, "current_node": 5, "direction": 1, "abs_arr": 100.0, "current_delay": 10.0, "seq": 0, "type": "Superfast", "priority": 3, "weather": "clear", "train": "A"},
        {"run_id": 2, "current_node": 5, "direction": 1, "abs_arr": 120.0, "current_delay": 50.0, "seq": 0, "type": "Superfast", "priority": 3, "weather": "clear", "train": "B"},
        {"run_id": 3, "current_node": 5, "direction": 1, "abs_arr": 150.0, "current_delay": 70.0, "seq": 0, "type": "Superfast", "priority": 3, "weather": "clear", "train": "C"},
    ]
    df = pd.DataFrame(rows)
    feat_df = extract_observation_features(df)

    # Train A has no prior arrival at node 5 -> lead delay must be NaN
    assert np.isnan(feat_df.loc[0, "station_lead_delay"])

    # Train B arrived at 120: preceding arrival was Train A (delay 10.0, gap 20.0m)
    assert feat_df.loc[1, "station_lead_delay"] == pytest.approx(10.0)
    assert feat_df.loc[1, "station_lead_gap_min"] == pytest.approx(20.0)

    # Train C arrived at 150: preceding arrival was Train B (delay 50.0, gap 30.0m), NOT Train A
    assert feat_df.loc[2, "station_lead_delay"] == pytest.approx(50.0)
    assert feat_df.loc[2, "station_lead_gap_min"] == pytest.approx(30.0)
