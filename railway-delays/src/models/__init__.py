"""RailCast Phase 6: End-to-End Downstream Delay Prediction.

Public API:
- FeaturePipeline, extract_observation_features, FEATURE_COLUMNS
- compute_downstream_targets, build_dataset_split, prepare_all_splits, get_horizon_subset
- PersistencePredictor, HistoricalPredictor
- create_regressor, train_horizon_model, train_full_pipeline
- calc_metrics, evaluate_predictions, evaluate_suite, plot_model_comparison
- DelayPredictor
- GNNRegressor
"""
from src.models.artifacts import load_model_bundle, save_model_bundle
from src.models.baseline import HistoricalPredictor, PersistencePredictor
from src.models.dataset import (
    HORIZONS,
    build_dataset_split,
    compute_downstream_targets,
    get_horizon_subset,
    prepare_all_splits,
)
from src.models.evaluate import (
    calc_metrics,
    evaluate_predictions,
    evaluate_suite,
    plot_actual_vs_predicted,
    plot_error_distributions,
    plot_model_comparison,
    save_metrics_json,
)
from src.models.features import (
    FEATURE_COLUMNS,
    FeaturePipeline,
    extract_observation_features,
)
from src.models.gnn import GNNRegressor
from src.models.predict import DelayPredictor
from src.models.train import create_regressor, train_full_pipeline, train_horizon_model

__all__ = [
    "FEATURE_COLUMNS",
    "FeaturePipeline",
    "extract_observation_features",
    "HORIZONS",
    "compute_downstream_targets",
    "build_dataset_split",
    "prepare_all_splits",
    "get_horizon_subset",
    "PersistencePredictor",
    "HistoricalPredictor",
    "create_regressor",
    "train_horizon_model",
    "train_full_pipeline",
    "calc_metrics",
    "evaluate_predictions",
    "evaluate_suite",
    "plot_actual_vs_predicted",
    "plot_error_distributions",
    "plot_model_comparison",
    "save_metrics_json",
    "save_model_bundle",
    "load_model_bundle",
    "DelayPredictor",
    "GNNRegressor",
]
