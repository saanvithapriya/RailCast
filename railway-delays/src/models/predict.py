"""Inference interface for Phase 6 downstream delay prediction models.

Provides a clean, standalone Python API to load trained model artifacts and generate
predictions for any or all supported forecast horizons (1-6h).

Designed for direct integration into Phase 7 Streamlit dashboard.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.models.artifacts import DEFAULT_MODEL_DIR, load_model_bundle
from src.models.features import FeaturePipeline

HORIZONS = [1, 2, 3, 4, 5, 6]


class DelayPredictor:
    """Production inference interface for trained delay forecasting models."""

    def __init__(
        self,
        models: Dict[int, Any],
        preprocessor: FeaturePipeline,
        metadata: Dict[str, Any],
    ):
        self.models = models
        self.preprocessor = preprocessor
        self.metadata = metadata
        self.model_type = metadata.get("model_type", "Unknown")
        self.available_horizons = sorted(list(models.keys()))

    @classmethod
    def load(cls, bundle_dir: Path | str = DEFAULT_MODEL_DIR) -> "DelayPredictor":
        """Load a trained model bundle from disk."""
        bundle = load_model_bundle(bundle_dir)
        return cls(
            models=bundle["models"],
            preprocessor=bundle["preprocessor"],
            metadata=bundle["metadata"],
        )

    def _normalize_input(self, input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """Validate and normalize user input into a standardized DataFrame."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(f"input_data must be a DataFrame or Dict, got {type(input_data)}")

        if len(df) == 0:
            raise ValueError("Input data contains 0 rows.")

        # Alias resolution for operational inputs
        if "current_delay" not in df.columns:
            if "arrival_delay" in df.columns:
                df["current_delay"] = df["arrival_delay"].astype(float)
            elif "delay" in df.columns:
                df["current_delay"] = df["delay"].astype(float)
            else:
                raise ValueError("Input data must contain 'current_delay' (or 'arrival_delay').")

        if "current_node" not in df.columns:
            if "node" in df.columns:
                df["current_node"] = df["node"].astype(int)
            else:
                # Default to node 0 if neither node nor current_node given
                df["current_node"] = 0

        return df

    def predict(
        self,
        input_data: Union[pd.DataFrame, Dict[str, Any]],
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Make delay predictions for a single forecast horizon h (in hours)."""
        if horizon not in self.available_horizons:
            raise ValueError(f"Requested horizon {horizon}h is not in available horizons {self.available_horizons}.")

        df = self._normalize_input(input_data)
        model = self.models[horizon]

        # Transform features using the learned training pipeline
        X = self.preprocessor.transform(df)
        preds = model.predict(X)

        result = pd.DataFrame({
            "predicted_delay_min": np.round(preds, 2),
            "horizon_hours": horizon,
            "units": "minutes",
            "model_type": self.model_type,
            "is_simulated": True,
        })

        if "train" in df.columns:
            result.insert(0, "train", df["train"].values)
        if "station" in df.columns:
            result.insert(1, "station", df["station"].values)

        return result

    def predict_all_horizons(
        self,
        input_data: Union[pd.DataFrame, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Generate downstream predictions across all six supported horizons (1h to 6h)."""
        df = self._normalize_input(input_data)
        X = self.preprocessor.transform(df)

        horizon_dfs = []
        for h in self.available_horizons:
            model = self.models[h]
            preds = model.predict(X)
            h_df = pd.DataFrame({
                "predicted_delay_min": np.round(preds, 2),
                "horizon_hours": h,
                "units": "minutes",
                "model_type": self.model_type,
                "is_simulated": True,
            })
            if "train" in df.columns:
                h_df.insert(0, "train", df["train"].values)
            if "station" in df.columns:
                h_df.insert(1, "station", df["station"].values)
            if "current_delay" in df.columns:
                h_df.insert(2, "current_delay_min", df["current_delay"].values)
            horizon_dfs.append(h_df)

        return pd.concat(horizon_dfs, ignore_index=True)
