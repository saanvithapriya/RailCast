"""Feature construction and preprocessing pipeline for Phase 6 delay prediction.

All features represent information strictly observable at prediction time t:
- Temporal: scheduled arrival time of day, cyclic encoding, day of week, month, season, corridor elapsed time.
- Train & section: current observed delay, previous departure delay, recent delay change, rolling delay,
  train type/priority, stop progress, station node, junction flag, distance along chain, direction.
- Network & corridor: delay of preceding train at this station (within 60m gap), station historical delay.
- Environmental: observable weather condition (clear, fog, rain) on the prediction date.

Leakage control:
- FeaturePipeline fits all statistics (node historical delay, imputers, scalers) strictly on training data.
- Future observations, hidden causes (cause_code, cause_train, root_event_id, primary_delay_injected),
  groundtruth cascades, and future weather observations are strictly excluded.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.eda import SEASONS, season_of
from src.graph.timetable import TYPES

TRAIN_TYPE_MAP = {t: i for i, t in enumerate(TYPES)}
SEASON_MAP = {s: i for i, s in enumerate(SEASONS)}

FEATURE_COLUMNS = [
    # Train / route features
    "train_type_code",
    "priority",
    "direction",
    "current_node",
    "is_junction",
    "km_from_new_delhi",
    "seq",
    "elapsed_corridor_min",
    "progress_ratio",
    # Train delay dynamics (observable at t)
    "current_delay",
    "prev_dep_delay",
    "delay_change_recent",
    "rolling_mean_delay_3",
    # Station & local network traffic (observable at t)
    "station_lead_delay",
    "station_lead_gap_min",
    "node_hist_mean_delay",
    "node_hist_p90_delay",
    # Temporal & seasonal
    "sched_hour",
    "sched_min",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "month",
    "season_code",
    # Weather (observable at date/time t)
    "is_fog",
    "is_rain",
]


def extract_observation_features(
    df: pd.DataFrame,
    nodes_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Extract raw observable feature columns from arrival observations.
    
    Expects df with columns:
    ['train', 'date', 'seq', 'station', 'scheduled_arrival', 'actual_arrival',
     'arrival_delay', 'departure_delay', 'abs_arr', 'run_id', 'node', 'direction',
     'type', 'priority'] plus optional 'weather', 'prev_dep_delay'.
    """
    out = df.copy()

    # Ensure required base columns exist
    if "current_delay" not in out.columns and "arrival_delay" in out.columns:
        out["current_delay"] = out["arrival_delay"].astype("float32")

    # Node / station attributes
    if "current_node" not in out.columns:
        if "node" in out.columns:
            out["current_node"] = out["node"].astype("int16")
        else:
            out["current_node"] = 0

    if "direction" not in out.columns:
        out["direction"] = 1

    if nodes_df is not None:
        node_map = nodes_df.set_index("node") if "node" in nodes_df.columns else nodes_df
        if "is_junction" in node_map.columns and "is_junction" not in out.columns:
            out["is_junction"] = out["current_node"].map(node_map["is_junction"]).fillna(0).astype("int8")
        if "km_from_new_delhi" in node_map.columns and "km_from_new_delhi" not in out.columns:
            out["km_from_new_delhi"] = out["current_node"].map(node_map["km_from_new_delhi"]).fillna(0.0).astype("float32")
    else:
        if "is_junction" not in out.columns:
            out["is_junction"] = 0
        if "km_from_new_delhi" not in out.columns:
            out["km_from_new_delhi"] = 0.0

    # Categorical encodings
    if "type" not in out.columns:
        out["type"] = "other"

    if "train_type_code" not in out.columns:
        out["train_type_code"] = out["type"].astype(str).map(TRAIN_TYPE_MAP).fillna(len(TYPES) - 1).astype("int8")

    if "priority" not in out.columns:
        out["priority"] = 4

    # Temporal features
    if "sched_hour" not in out.columns:
        if "scheduled_arrival" in out.columns:
            out["sched_hour"] = ((out["scheduled_arrival"].astype("float64") % 1440) // 60).astype("int8")
            out["sched_min"] = (out["scheduled_arrival"].astype("float64") % 60).astype("int8")
        else:
            out["sched_hour"] = 12
            out["sched_min"] = 0
    elif "sched_min" not in out.columns:
        if "scheduled_arrival" in out.columns:
            out["sched_min"] = (out["scheduled_arrival"].astype("float64") % 60).astype("int8")
        else:
            out["sched_min"] = 0

    rad = 2.0 * np.pi * out["sched_hour"].astype("float32") / 24.0
    out["hour_sin"] = np.sin(rad).astype("float32")
    out["hour_cos"] = np.cos(rad).astype("float32")

    if "day_of_week" not in out.columns:
        if "dow" in out.columns:
            out["day_of_week"] = out["dow"].astype("int8")
        elif "date" in out.columns:
            out["day_of_week"] = pd.to_datetime(out["date"]).dt.dayofweek.astype("int8")
        else:
            out["day_of_week"] = 0

    if "month" not in out.columns:
        if "date" in out.columns:
            out["month"] = pd.to_datetime(out["date"]).dt.month.astype("int8")
        else:
            out["month"] = 1

    if "season_code" not in out.columns:
        if "season" in out.columns:
            out["season_code"] = out["season"].astype(str).map(SEASON_MAP).fillna(0).astype("int8")
        else:
            seasons = season_of(out["month"].to_numpy())
            out["season_code"] = pd.Series(seasons, index=out.index).map(SEASON_MAP).fillna(0).astype("int8")

    # Run dynamics: corridor elapsed time and progress
    if "run_id" in out.columns and "abs_arr" in out.columns:
        g = out.groupby("run_id", sort=False)
        first_arr = g["abs_arr"].transform("first")
        max_seq = g["seq"].transform("max")
        out["elapsed_corridor_min"] = (out["abs_arr"] - first_arr).clip(lower=0.0).astype("float32")
        out["progress_ratio"] = (out["seq"].astype("float32") / max_seq.clip(lower=1.0)).astype("float32")
    else:
        if "elapsed_corridor_min" not in out.columns:
            out["elapsed_corridor_min"] = 0.0
        if "progress_ratio" not in out.columns:
            out["progress_ratio"] = 0.5

    # Delay dynamics
    if "prev_dep_delay" not in out.columns:
        if "run_id" in out.columns and "departure_delay" in out.columns:
            out["prev_dep_delay"] = out.groupby("run_id", sort=False)["departure_delay"].shift()
        else:
            out["prev_dep_delay"] = out["current_delay"]

    if "delay_change_recent" not in out.columns:
        out["delay_change_recent"] = (out["current_delay"] - out["prev_dep_delay"]).fillna(0.0).astype("float32")

    if "rolling_mean_delay_3" not in out.columns:
        if "run_id" in out.columns and "current_delay" in out.columns:
            # Shifted rolling mean over past stops up to current stop
            out["rolling_mean_delay_3"] = (
                out.groupby("run_id", sort=False)["current_delay"]
                .rolling(3, min_periods=1)
                .mean()
                .reset_index(drop=True)
                .astype("float32")
            )
        else:
            out["rolling_mean_delay_3"] = out["current_delay"]

    # Station leading train traffic (observable past arrival at same node and direction)
    if "station_lead_delay" not in out.columns:
        if "abs_arr" in out.columns:
            d_sorted = out.sort_values(["current_node", "direction", "abs_arr"])
            key = d_sorted["current_node"].to_numpy(dtype=np.int64) * 4 + (d_sorted["direction"].to_numpy() < 0).astype(np.int64)
            prev_t = d_sorted["abs_arr"].groupby(key).shift()
            prev_d = d_sorted["current_delay"].groupby(key).shift()
            gap = d_sorted["abs_arr"] - prev_t

            valid_lead = (gap >= 0.0) & (gap <= 60.0)
            lead_d = np.where(valid_lead, prev_d, np.nan)
            lead_gap = np.where(valid_lead, gap, 120.0)

            out.loc[d_sorted.index, "station_lead_delay"] = lead_d.astype("float32")
            out.loc[d_sorted.index, "station_lead_gap_min"] = lead_gap.astype("float32")
        else:
            out["station_lead_delay"] = np.nan
            out["station_lead_gap_min"] = 120.0

    # Weather flags
    if "is_fog" not in out.columns:
        if "weather" in out.columns:
            out["is_fog"] = (out["weather"].astype(str) == "fog").astype("int8")
            out["is_rain"] = (out["weather"].astype(str) == "rain").astype("int8")
        else:
            out["is_fog"] = 0
            out["is_rain"] = 0

    return out


class FeaturePipeline:
    """Learned preprocessing and feature transformation pipeline.
    
    Fits node delay baselines and default imputation statistics strictly on training data.
    """

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.node_hist_mean: Dict[int, float] = {}
        self.node_hist_p90: Dict[int, float] = {}
        self.overall_mean_delay: float = 0.0
        self.impute_values: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        """Compute summary statistics exclusively from the training dataframe."""
        df = extract_observation_features(train_df)

        self.overall_mean_delay = float(df["current_delay"].dropna().mean())

        # Historical delay per station node
        node_stats = df.groupby("current_node")["current_delay"].agg(["mean", lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) > 0 else 0.0])
        self.node_hist_mean = node_stats["mean"].to_dict()
        self.node_hist_p90 = node_stats["<lambda_0>"].to_dict()

        # Compute default imputation values
        self.impute_values = {
            "current_delay": self.overall_mean_delay,
            "prev_dep_delay": self.overall_mean_delay,
            "delay_change_recent": 0.0,
            "rolling_mean_delay_3": self.overall_mean_delay,
            "station_lead_delay": self.overall_mean_delay,
            "station_lead_gap_min": 120.0,
            "node_hist_mean_delay": self.overall_mean_delay,
            "node_hist_p90_delay": self.overall_mean_delay,
            "elapsed_corridor_min": float(df["elapsed_corridor_min"].median() if "elapsed_corridor_min" in df else 0.0),
            "progress_ratio": 0.5,
            "is_junction": 0.0,
            "km_from_new_delhi": 0.0,
            "is_fog": 0.0,
            "is_rain": 0.0,
        }

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform observation DataFrame into numeric feature matrix X."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted on training data before calling transform().")

        features_df = extract_observation_features(df)

        # Attach historical station statistics learned during fit()
        features_df["node_hist_mean_delay"] = (
            features_df["current_node"].map(self.node_hist_mean).fillna(self.overall_mean_delay).astype("float32")
        )
        features_df["node_hist_p90_delay"] = (
            features_df["current_node"].map(self.node_hist_p90).fillna(self.overall_mean_delay).astype("float32")
        )

        # Impute missing values with learned training statistics
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = self.impute_values.get(col, 0.0)
            else:
                default_val = self.impute_values.get(col, 0.0)
                features_df[col] = features_df[col].fillna(default_val)

        return features_df[self.feature_columns].to_numpy(dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        return list(self.feature_columns)
