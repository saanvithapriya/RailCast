"""Unit tests for dataset generation, target alignment, and feature extraction."""
import numpy as np
import pandas as pd
import pytest

from src.models.dataset import HORIZONS, compute_downstream_targets, get_horizon_subset
from src.models.features import FEATURE_COLUMNS, FeaturePipeline, extract_observation_features


@pytest.fixture
def synthetic_runs():
    """Create 2 synthetic train runs with known timestamps and delays."""
    rows = []
    # Run 1: 8 stops, spaced ~60 min apart (0, 60, 120, 180, 240, 300, 360, 420 min)
    for i in range(8):
        t = float(i * 60)
        rows.append({
            "run_id": 1,
            "train": "12301",
            "date": "2024-05-01",
            "station": f"S{i}",
            "seq": i,
            "node": i,
            "abs_arr": t,
            "actual_arrival": t % 1440.0,
            "actual_departure": (t + 2) % 1440.0,
            "scheduled_arrival": t % 1440.0,
            "scheduled_departure": (t + 2) % 1440.0,
            "arrival_delay": float(10 + i * 2),  # delays: 10, 12, 14, 16, 18, 20, 22, 24
            "departure_delay": float(10 + i * 2),
            "type": "Rajdhani",
            "priority": 1,
            "direction": 1,
            "weather": "clear",
            "halt": i in (0, 3, 7),
        })

    # Run 2: Short run with only 3 stops (terminates at stop 2, t=120)
    for i in range(3):
        t = float(i * 60)
        rows.append({
            "run_id": 2,
            "train": "64001",
            "date": "2024-05-01",
            "station": f"P{i}",
            "seq": i,
            "node": i + 20,
            "abs_arr": t,
            "actual_arrival": t % 1440.0,
            "actual_departure": (t + 1) % 1440.0,
            "scheduled_arrival": t % 1440.0,
            "scheduled_departure": (t + 1) % 1440.0,
            "arrival_delay": float(5.0),
            "departure_delay": float(5.0),
            "type": "Passenger",
            "priority": 5,
            "direction": 1,
            "weather": "fog",
            "halt": True,
        })

    return pd.DataFrame(rows)


def test_target_generation_all_six_horizons(synthetic_runs):
    """Verify target generation accurately aligns downstream arrival delay for horizons 1-6h."""
    df_targets = compute_downstream_targets(synthetic_runs, horizons=HORIZONS, tol_min=35.0)

    for h in HORIZONS:
        assert f"target_delay_{h}h" in df_targets.columns

    # For Run 1 at stop 0 (t=0, delay=10):
    # stop 1 is at t=60 (delay=12) -> horizon 1h target should be 12.0
    r1_stop0 = df_targets[(df_targets.run_id == 1) & (df_targets.seq == 0)].iloc[0]
    assert r1_stop0["target_delay_1h"] == pytest.approx(12.0)
    # stop 2 is at t=120 (delay=14) -> horizon 2h target should be 14.0
    assert r1_stop0["target_delay_2h"] == pytest.approx(14.0)
    # stop 6 is at t=360 (delay=22) -> horizon 6h target should be 22.0
    assert r1_stop0["target_delay_6h"] == pytest.approx(22.0)


def test_terminating_train_has_nan_for_unavailable_horizons(synthetic_runs):
    """Trains that terminate or have no observation within tolerance must yield NaN targets."""
    df_targets = compute_downstream_targets(synthetic_runs, horizons=HORIZONS, tol_min=35.0)

    # Run 2 only lasts 120 minutes (stops at 0, 60, 120).
    # From stop 0 (t=0): 1h (t=60) exists, 2h (t=120) exists, but 3h..6h do NOT exist.
    r2_stop0 = df_targets[(df_targets.run_id == 2) & (df_targets.seq == 0)].iloc[0]
    assert r2_stop0["target_delay_1h"] == pytest.approx(5.0)
    assert r2_stop0["target_delay_2h"] == pytest.approx(5.0)
    assert np.isnan(r2_stop0["target_delay_3h"])
    assert np.isnan(r2_stop0["target_delay_4h"])
    assert np.isnan(r2_stop0["target_delay_5h"])
    assert np.isnan(r2_stop0["target_delay_6h"])

    # From stop 2 (t=120, terminus): NO future arrivals exist at all!
    r2_terminus = df_targets[(df_targets.run_id == 2) & (df_targets.seq == 2)].iloc[0]
    for h in HORIZONS:
        assert np.isnan(r2_terminus[f"target_delay_{h}h"])


def test_get_horizon_subset(synthetic_runs):
    """get_horizon_subset must filter strictly to rows where the target is not null."""
    df_targets = compute_downstream_targets(synthetic_runs, horizons=HORIZONS)
    sub1 = get_horizon_subset(df_targets, horizon=1)
    assert sub1["target_delay_1h"].isna().sum() == 0
    assert len(sub1) > 0

    sub6 = get_horizon_subset(df_targets, horizon=6)
    assert sub6["target_delay_6h"].isna().sum() == 0
    # Run 2 has 0 valid rows for 6h; Run 1 only has stop 0 and 1 with >=6h remaining
    assert len(sub6) < len(sub1)


def test_feature_pipeline_fit_and_transform(synthetic_runs):
    """Verify FeaturePipeline extracts all schema features and fits statistics strictly on train."""
    pipeline = FeaturePipeline()
    assert not pipeline.is_fitted

    with pytest.raises(RuntimeError):
        pipeline.transform(synthetic_runs)

    pipeline.fit(synthetic_runs)
    assert pipeline.is_fitted

    X = pipeline.transform(synthetic_runs)
    assert X.shape == (len(synthetic_runs), len(FEATURE_COLUMNS))
    assert not np.isnan(X).any()  # Imputers must guarantee no NaNs in output matrix
