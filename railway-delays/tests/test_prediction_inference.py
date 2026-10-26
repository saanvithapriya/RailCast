"""Unit tests for the DelayPredictor inference API."""
import numpy as np
import pandas as pd
import pytest

from src.models.artifacts import save_model_bundle
from src.models.features import FeaturePipeline
from src.models.predict import DelayPredictor
from src.models.train import create_regressor


@pytest.fixture
def mock_bundle(tmp_path):
    """Create a minimal saved model bundle for inference testing."""
    bundle_dir = tmp_path / "mock_model"

    # Fit a tiny dummy pipeline
    dummy_df = pd.DataFrame([{
        "run_id": 1,
        "train": "12001",
        "station": "NDLS",
        "current_node": 0,
        "current_delay": 15.0,
        "arrival_delay": 15.0,
        "departure_delay": 15.0,
        "seq": 0,
        "abs_arr": 600.0,
        "type": "Shatabdi",
        "priority": 2,
        "weather": "clear",
    }])
    pipeline = FeaturePipeline().fit(dummy_df)
    X = pipeline.transform(dummy_df)

    models = {}
    for h in [1, 2, 3, 4, 5, 6]:
        reg = create_regressor(model_type="rf", horizon=h, n_estimators=2, max_depth=2)
        reg.fit(X, np.array([20.0 + h]))
        models[h] = reg

    metadata = {
        "model_type": "RandomForest",
        "feature_names": pipeline.get_feature_names(),
        "random_state": 42,
    }
    save_model_bundle(bundle_dir, models, pipeline, metadata)
    return bundle_dir


def test_predictor_load(mock_bundle):
    """Verify loading from valid directory and error handling for missing directory."""
    predictor = DelayPredictor.load(mock_bundle)
    assert predictor.model_type == "RandomForest"
    assert predictor.available_horizons == [1, 2, 3, 4, 5, 6]

    with pytest.raises(FileNotFoundError):
        DelayPredictor.load(mock_bundle / "non_existent_dir")


def test_predict_single_dict_input(mock_bundle):
    """Test prediction from a single dictionary input."""
    predictor = DelayPredictor.load(mock_bundle)

    inp = {
        "train": "12001",
        "current_delay": 12.5,
        "current_node": 0,
        "station": "NDLS",
        "weather": "clear",
    }
    out = predictor.predict(inp, horizon=1)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1
    assert "predicted_delay_min" in out.columns
    assert out.iloc[0]["horizon_hours"] == 1
    assert bool(out.iloc[0]["is_simulated"]) is True


def test_predict_all_horizons(mock_bundle):
    """Test multi-horizon predictions across 1h to 6h."""
    predictor = DelayPredictor.load(mock_bundle)

    inp = pd.DataFrame([{
        "train": "12001",
        "current_delay": 10.0,
        "current_node": 0,
        "station": "NDLS",
    }])
    out = predictor.predict_all_horizons(inp)
    assert len(out) == 6
    assert list(out["horizon_hours"]) == [1, 2, 3, 4, 5, 6]
    assert (out["units"] == "minutes").all()


def test_predict_validation_errors(mock_bundle):
    """Verify input validation catches missing delay and invalid horizon."""
    predictor = DelayPredictor.load(mock_bundle)

    # Missing current_delay (and arrival_delay)
    with pytest.raises(ValueError, match="current_delay"):
        predictor.predict({"train": "12001", "station": "NDLS"}, horizon=1)

    # Unsupported horizon (e.g. 7h or 0h)
    with pytest.raises(ValueError, match="Requested horizon"):
        predictor.predict({"train": "12001", "current_delay": 5.0}, horizon=7)
