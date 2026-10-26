"""Baseline predictors for Phase 6 downstream delay prediction.

Implements:
1. Persistence baseline: Predicts latest observable arrival delay as the future delay across all horizons.
2. Historical baseline: Training-only historical lookup hierarchy (train x station, station x hour, overall).
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class PersistencePredictor:
    """Persistence baseline: forecasts that a train's future delay equals its current observed delay."""

    def __init__(self):
        self.name = "Persistence"

    def fit(self, train_df: Optional[pd.DataFrame] = None) -> "PersistencePredictor":
        """No training needed for persistence; interface consistency only."""
        return self

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Forecast downstream delay at horizon h as current_delay."""
        if "current_delay" in df.columns:
            return df["current_delay"].to_numpy(dtype=np.float32)
        elif "arrival_delay" in df.columns:
            return df["arrival_delay"].to_numpy(dtype=np.float32)
        else:
            raise KeyError("Neither 'current_delay' nor 'arrival_delay' found in input DataFrame.")


class HistoricalPredictor:
    """Historical baseline: predicts downstream delay using historical training averages.
    
    Uses a 3-level fallback hierarchy fitted strictly on the training set:
    1. Mean delay for (train, current_node) at horizon h.
    2. Mean delay for (current_node, sched_hour) at horizon h.
    3. Global mean delay at horizon h in the training set.
    """

    def __init__(self):
        self.name = "Historical"
        self.train_node_means: Dict[Tuple[str, int, int], float] = {}
        self.node_hour_means: Dict[Tuple[int, int, int], float] = {}
        self.horizon_means: Dict[int, float] = {}
        self.global_mean: float = 0.0
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "HistoricalPredictor":
        """Fit historical averages exclusively on training data."""
        df = train_df.copy()
        if "current_node" not in df.columns:
            if "node" in df.columns:
                df["current_node"] = df["node"].astype(int)
            else:
                df["current_node"] = 0

        train_str = df["train"].astype(str)
        node_arr = df["current_node"].astype(int)
        hour_arr = df["sched_hour"].astype(int) if "sched_hour" in df.columns else pd.Series(0, index=df.index, dtype=int)

        all_targets = []
        for h in (1, 2, 3, 4, 5, 6):
            t_col = f"target_delay_{h}h"
            if t_col not in df.columns:
                continue

            valid = df[t_col].notna()
            sub_y = df.loc[valid, t_col].to_numpy(dtype=np.float64)
            if len(sub_y) == 0:
                continue

            all_targets.append(sub_y)
            self.horizon_means[h] = float(np.mean(sub_y))

            # Level 1: (train, node, horizon)
            sub_train = train_str.loc[valid]
            sub_node = node_arr.loc[valid]
            g1 = pd.DataFrame({"y": sub_y, "train": sub_train, "node": sub_node}).groupby(["train", "node"])["y"].mean()
            for (tr, nd), val in g1.items():
                self.train_node_means[(str(tr), int(nd), h)] = float(val)

            # Level 2: (node, hour, horizon)
            sub_hour = hour_arr.loc[valid]
            g2 = pd.DataFrame({"y": sub_y, "node": sub_node, "hour": sub_hour}).groupby(["node", "hour"])["y"].mean()
            for (nd, hr), val in g2.items():
                self.node_hour_means[(int(nd), int(hr), h)] = float(val)

        if all_targets:
            self.global_mean = float(np.mean(np.concatenate(all_targets)))
        else:
            self.global_mean = 0.0

        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Predict downstream delay at horizon h for rows in df."""
        if not self.is_fitted:
            raise RuntimeError("HistoricalPredictor must be fitted before predict().")

        train_str = df["train"].astype(str).to_numpy() if "train" in df.columns else np.array(["unknown"] * len(df))
        if "current_node" in df.columns:
            node_arr = df["current_node"].astype(int).to_numpy()
        elif "node" in df.columns:
            node_arr = df["node"].astype(int).to_numpy()
        else:
            node_arr = np.zeros(len(df), dtype=int)
        hour_arr = df["sched_hour"].astype(int).to_numpy() if "sched_hour" in df.columns else np.zeros(len(df), dtype=int)

        h_mean = self.horizon_means.get(horizon, self.global_mean)
        preds = np.full(len(df), h_mean, dtype=np.float32)

        for i in range(len(df)):
            key1 = (train_str[i], node_arr[i], horizon)
            if key1 in self.train_node_means:
                preds[i] = self.train_node_means[key1]
            else:
                key2 = (node_arr[i], hour_arr[i], horizon)
                if key2 in self.node_hour_means:
                    preds[i] = self.node_hour_means[key2]

        return preds
