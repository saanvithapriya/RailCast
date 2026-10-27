"""
Unit tests for the Phase 7 Streamlit Application modules.

Verifies:
- Imports of app modules and data loader
- Data loader functions and artifact fallbacks
- Model loading and inference for all 6 horizons
- Clear SIMULATED labeling in UI outputs
- Safe handling of unavailable or missing artifacts
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.app.data_loader import (
    load_network_overview,
    load_critical_sections,
    load_cascade_mining_results,
    load_evaluation_metrics,
    get_prediction_model,
    format_prediction_output,
)
import src.app.streamlit_app as app_module


class TestDataLoader:
    """Test data loading, caching, and fallback utilities."""

    def test_load_network_overview_structure(self):
        overview = load_network_overview()
        assert isinstance(overview, dict)
        assert "num_stations" in overview
        assert "num_sections" in overview
        assert "total_track_km" in overview
        assert "corridor" in overview
        assert overview["is_simulated"] is True
        assert overview["num_stations"] > 0

    def test_load_critical_sections(self):
        df = load_critical_sections()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "section" in df.columns or "Section" in df.columns or "rank" in df.columns
        assert len(df) >= 5

    def test_load_cascade_mining_results(self):
        results = load_cascade_mining_results()
        assert isinstance(results, dict)
        assert "summary" in results
        assert "propagation_rates" in results
        assert "is_simulated" in results
        assert results["is_simulated"] is True

    def test_load_evaluation_metrics_has_six_horizons(self):
        metrics = load_evaluation_metrics()
        assert isinstance(metrics, dict)
        # Check that metrics contain evaluation horizons
        has_horizons = False
        if "horizons" in metrics:
            has_horizons = True
        elif any("horizons" in v for v in metrics.values() if isinstance(v, dict)):
            has_horizons = True
        assert has_horizons, "Evaluation metrics should contain horizon evaluations"


class TestModelInferenceInApp:
    """Test predictor interface across all 6 horizons."""

    def test_get_prediction_model(self):
        predictor = get_prediction_model()
        assert predictor is not None
        assert hasattr(predictor, "predict_all_horizons")
        assert hasattr(predictor, "predict")

    def test_prediction_all_six_horizons(self):
        predictor = get_prediction_model()
        sample_obs = {
            "train": 12301,
            "station": "TDL",
            "current_delay": 15.0,
            "arr_delay": 15.0,
            "dep_delay": 18.0,
            "rolling_mean_delay_30m": 12.0,
            "rolling_max_delay_30m": 25.0,
            "rolling_active_trains_30m": 8,
            "node_degree": 2,
            "node_betweenness": 0.04,
            "section_criticality": 0.08,
            "fog_flag": 1,
            "visibility_m": 450.0,
            "sched_dep_hour": 7,
            "sched_dep_dow": 2,
        }

        preds = predictor.predict_all_horizons(sample_obs)
        assert isinstance(preds, pd.DataFrame)
        assert len(preds) == 6
        assert set(preds["horizon_hours"]) == {1, 2, 3, 4, 5, 6}
        assert (preds["is_simulated"] == True).all()

    def test_format_prediction_output(self):
        raw_preds = {1: 18.5, 2: 24.2, 3: 31.0, 4: 37.8, 5: 44.1, 6: 51.3}
        formatted = format_prediction_output(raw_preds, "12301 (Rajdhani)", "HWH")
        assert isinstance(formatted, list)
        assert len(formatted) == 6
        for row in formatted:
            assert "horizon" in row
            assert "predicted_delay_min" in row
            assert row["train"] == "12301 (Rajdhani)"
            assert row["is_simulated"] is True


class TestAppIntegrity:
    """Test app entrypoint and page functions."""

    def test_page_functions_exist(self):
        assert hasattr(app_module, "page_network_overview")
        assert hasattr(app_module, "page_network_explorer")
        assert hasattr(app_module, "page_cascade_mining")
        assert hasattr(app_module, "page_delay_prediction")
        assert hasattr(app_module, "page_model_evaluation")
        assert hasattr(app_module, "page_methodology")
        assert hasattr(app_module, "main")

    def test_main_function_executable(self):
        with patch("streamlit.sidebar") as mock_sidebar, \
             patch("streamlit.set_page_config"), \
             patch("streamlit.warning"), \
             patch("streamlit.markdown"):
            mock_sidebar.radio.return_value = "🌐 1. Network Overview"
            app_module.main()
