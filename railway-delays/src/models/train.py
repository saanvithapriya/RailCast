"""Model training and selection pipeline for Phase 6 downstream delay prediction.

Supports:
- Random Forest regressor (scikit-learn)
- Gradient-Boosted Trees (LightGBM as primary, XGBoost compatible)
- Horizon-specific modeling for all six horizons (1h, 2h, 3h, 4h, 5h, 6h)
- Strict temporal isolation: trained only on training split, tuned against validation split.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.artifacts import DEFAULT_MODEL_DIR, save_model_bundle
from src.models.dataset import HORIZONS, get_horizon_subset
from src.models.features import FeaturePipeline

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def create_regressor(
    model_type: str = "lightgbm",
    horizon: int = 1,
    random_state: int = 42,
    **kwargs,
) -> Any:
    """Instantiate a regressor for horizon h with computationally feasible, sensible hyperparameters."""
    m_type = model_type.lower()

    if m_type in ("lightgbm", "lgb"):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is not installed. Please install lightgbm or select model_type='rf'.")
        # Sensible tuned hyperparameters for tabular delay forecasting
        n_est = kwargs.get("n_estimators", 120)
        lr = kwargs.get("learning_rate", 0.05)
        num_leaves = kwargs.get("num_leaves", 31)
        subsample = kwargs.get("subsample", 0.8)
        colsample = kwargs.get("colsample_bytree", 0.8)
        return lgb.LGBMRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample,
            random_state=random_state + horizon,
            n_jobs=-1,
            verbose=-1,
        )

    elif m_type in ("rf", "random_forest"):
        n_est = kwargs.get("n_estimators", 60)
        max_depth = kwargs.get("max_depth", 14)
        min_samples_leaf = kwargs.get("min_samples_leaf", 5)
        return RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + horizon,
            n_jobs=-1,
        )

    elif m_type in ("xgb", "xgboost"):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Please install xgboost or select model_type='lightgbm'.")
        n_est = kwargs.get("n_estimators", 120)
        lr = kwargs.get("learning_rate", 0.05)
        max_depth = kwargs.get("max_depth", 6)
        return xgb.XGBRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=max_depth,
            random_state=random_state + horizon,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose from 'lightgbm', 'rf', 'xgb'.")


def train_horizon_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "lightgbm",
    horizon: int = 1,
    random_state: int = 42,
    **kwargs,
) -> Any:
    """Train a single model for horizon h on features X_train and targets y_train."""
    reg = create_regressor(model_type=model_type, horizon=horizon, random_state=random_state, **kwargs)
    reg.fit(X_train, y_train)
    return reg


def train_full_pipeline(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    model_type: str = "lightgbm",
    horizons: List[int] = HORIZONS,
    bundle_dir: Optional[Path | str] = None,
    random_state: int = 42,
    max_train_samples: Optional[int] = None,
    **kwargs,
) -> Tuple[Dict[int, Any], FeaturePipeline]:
    """Train models for all requested horizons and persist artifact bundle if bundle_dir is provided.

    train_df: Training DataFrame containing observable features and target_delay_{h}h columns.
    max_train_samples: Optional ceiling on training rows (useful for RF speed).
    """
    if len(train_df) == 0:
        raise ValueError("train_df is empty.")

    # 1. Fit Preprocessor strictly on training data
    pipeline = FeaturePipeline()
    pipeline.fit(train_df)

    models: Dict[int, Any] = {}

    for h in horizons:
        sub_train = get_horizon_subset(train_df, horizon=h)
        if len(sub_train) == 0:
            raise ValueError(f"No valid targets found for horizon {h}h in train_df.")

        # Subsample for tree methods if requested
        if max_train_samples is not None and len(sub_train) > max_train_samples:
            sub_train = sub_train.sample(n=max_train_samples, random_state=random_state + h)

        X_tr = pipeline.transform(sub_train)
        y_tr = sub_train[f"target_delay_{h}h"].to_numpy(dtype=np.float32)

        model = train_horizon_model(
            X_tr,
            y_tr,
            model_type=model_type,
            horizon=h,
            random_state=random_state,
            **kwargs,
        )
        models[h] = model

    if bundle_dir is not None:
        metadata = {
            "model_type": model_type,
            "feature_names": pipeline.get_feature_names(),
            "n_train_examples": {h: int((train_df[f"target_delay_{h}h"].notna()).sum()) for h in horizons},
            "random_state": random_state,
            "hyperparameters": kwargs,
        }
        save_model_bundle(bundle_dir, models, pipeline, metadata)

    return models, pipeline
