"""Unit tests for baseline predictors, ML training, evaluation metrics, and serialization."""
import numpy as np
import pandas as pd
import pytest

from src.models.artifacts import load_model_bundle, save_model_bundle
from src.models.baseline import HistoricalPredictor, PersistencePredictor
from src.models.evaluate import calc_metrics
from src.models.features import FeaturePipeline
from src.models.gnn import GNNRegressor
from src.models.train import create_regressor, train_full_pipeline, train_horizon_model


@pytest.fixture
def train_val_fixtures():
    """Small synthetic train/val fixtures for fast unit testing."""
    np.random.seed(42)
    n_train, n_val = 150, 40

    def make_df(n, base_delay):
        rows = []
        for i in range(n):
            d = float(base_delay + np.random.uniform(-5, 15))
            rows.append({
                "run_id": i // 5,
                "train": str(12000 + (i % 4)),
                "date": "2024-03-01",
                "station": f"S{i%10}",
                "seq": i % 5,
                "node": i % 10,
                "abs_arr": float(i * 30),
                "scheduled_arrival": float((i * 30) % 1440),
                "scheduled_departure": float((i * 30 + 2) % 1440),
                "arrival_delay": d,
                "departure_delay": d,
                "current_delay": d,
                "type": "Superfast" if i % 2 == 0 else "Passenger",
                "priority": 3 if i % 2 == 0 else 5,
                "direction": 1 if i % 3 != 0 else -1,
                "weather": "clear" if i % 4 != 0 else "fog",
                "is_fog": 1 if i % 4 == 0 else 0,
                "is_rain": 0,
                # Create targets for all 6 horizons
                **{f"target_delay_{h}h": d + float(h * 1.5 + np.random.normal(0, 1)) for h in range(1, 7)},
            })
        return pd.DataFrame(rows)

    return make_df(n_train, 15.0), make_df(n_val, 20.0)


def test_persistence_baseline(train_val_fixtures):
    """Persistence baseline must return current_delay exactly."""
    train_df, val_df = train_val_fixtures
    model = PersistencePredictor()
    model.fit(train_df)

    preds = model.predict(val_df, horizon=1)
    np.testing.assert_allclose(preds, val_df["current_delay"].to_numpy(dtype=np.float32))


def test_historical_baseline(train_val_fixtures):
    """Historical baseline must fit training averages and provide sensible predictions."""
    train_df, val_df = train_val_fixtures
    model = HistoricalPredictor()
    model.fit(train_df)
    assert model.is_fitted

    preds = model.predict(val_df, horizon=2)
    assert len(preds) == len(val_df)
    assert not np.isnan(preds).any()
    # Forecasts should be in the realm of training targets (~15-25 min)
    assert 10.0 <= np.mean(preds) <= 30.0


def test_calc_metrics_calculation():
    """Verify MAE, RMSE, R2, and baseline improvement percentage calculations."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 38.0])  # errors: +2, -2, +3, -2 -> MAE = 2.25
    y_base = np.array([15.0, 25.0, 35.0, 45.0])  # errors: +5, +5, +5, +5 -> MAE = 5.0

    m = calc_metrics(y_true, y_pred, y_baseline=y_base)
    assert m["n"] == 4
    assert m["mae"] == pytest.approx(2.25)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9 + 4) / 4), abs=1e-3)
    assert m["r2"] > 0.9
    # Baseline improvement = (5.0 - 2.25) / 5.0 = 55.0%
    assert m["improvement_pct"] == pytest.approx(55.0)


def test_train_pipeline_and_artifact_bundle(train_val_fixtures, tmp_path):
    """Test full training pipeline with LightGBM, artifact saving, and loading."""
    train_df, val_df = train_val_fixtures
    bundle_dir = tmp_path / "models"

    models, pipeline = train_full_pipeline(
        train_df,
        val_df=val_df,
        model_type="lightgbm",
        horizons=[1, 2],
        bundle_dir=bundle_dir,
        n_estimators=10,
        random_state=42,
    )

    assert 1 in models and 2 in models
    assert pipeline.is_fitted
    assert (bundle_dir / "metadata.json").exists()
    assert (bundle_dir / "preprocessor.joblib").exists()
    assert (bundle_dir / "model_h1.joblib").exists()

    bundle = load_model_bundle(bundle_dir)
    loaded_m1 = bundle["models"][1]
    X_val = bundle["preprocessor"].transform(val_df)
    preds = loaded_m1.predict(X_val)
    assert len(preds) == len(val_df)


def test_random_forest_training_fixture(train_val_fixtures):
    """Test RandomForestRegressor training on small fixture."""
    train_df, val_df = train_val_fixtures
    models, pipeline = train_full_pipeline(
        train_df,
        model_type="rf",
        horizons=[1],
        n_estimators=5,
        max_depth=5,
        random_state=42,
    )
    X_val = pipeline.transform(val_df)
    preds = models[1].predict(X_val)
    assert len(preds) == len(val_df)
    assert not np.isnan(preds).any()


def test_gnn_regressor_fixture(train_val_fixtures):
    """Test GNNRegressor on small synthetic fixture."""
    train_df, val_df = train_val_fixtures
    pipeline = FeaturePipeline().fit(train_df)
    X_tr = pipeline.transform(train_df)
    y_tr = train_df["target_delay_1h"].to_numpy(dtype=np.float32)

    gnn = GNNRegressor(horizon=1, epochs=3, batch_size=32, random_state=42)
    gnn.fit(X_tr, y_tr)

    X_val = pipeline.transform(val_df)
    preds = gnn.predict(X_val)
    assert len(preds) == len(val_df)
    assert not np.isnan(preds).any()
