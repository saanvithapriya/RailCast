"""Forecasting dataset generation for Phase 6 downstream delay prediction.

Constructs training, validation, and test examples across all six forecast horizons (1-6h):
- Prediction unit: An eligible train arrival observation at a corridor station at observation time t.
- Target: Downstream arrival delay (in minutes) of the SAME train run at the station reached
  closest to t_now + h hours (within tolerance of +/- 35 minutes).
- Clean handling of boundary cases: trains that terminate or exit the corridor before t + h hours
  have no downstream record, yielding target = NaN (no synthetic or leaked targets).
- Split configuration is read directly from configs/splits.yaml via src/splits.py.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import splits
from src.analysis.eda import arrivals, load_window
from src.graph.station_graph import build_station_tables
from src.graph.timetable import load_timetable
from src.models.features import extract_observation_features

HORIZONS = [1, 2, 3, 4, 5, 6]
DEFAULT_TOL_MIN = 35.0
DEFAULT_DATA_DIR = Path("data/processed/prediction")


def compute_downstream_targets(
    arrivals_df: pd.DataFrame,
    horizons: List[int] = HORIZONS,
    tol_min: float = DEFAULT_TOL_MIN,
) -> pd.DataFrame:
    """Vectorized calculation of downstream arrival delay targets for all horizons.

    For each arrival row at t_now of a run, finds the arrival of the same run closest to
    t_now + 60*h minutes. If no arrival exists within +/- tol_min (e.g. train terminated or exited),
    the target for that horizon is NaN.
    """
    df = arrivals_df.sort_values(["run_id", "abs_arr"]).reset_index(drop=True)
    run = df["run_id"].to_numpy()
    t = df["abs_arr"].to_numpy(dtype="float64")
    key = run.astype("float64") * 1e7 + t
    n_rows = len(df)

    out = df.copy()

    for h in horizons:
        target_t = key + 60.0 * float(h)
        idx = np.searchsorted(key, target_t)
        best = np.full(n_rows, -1, dtype=np.int64)
        bestd = np.full(n_rows, np.inf, dtype=np.float64)

        for c in (idx - 1, idx):
            cc = np.clip(c, 0, n_rows - 1)
            dist = np.abs(t[cc] - (t + 60.0 * float(h)))
            good = (
                (c >= 0)
                & (c < n_rows)
                & (run[cc] == run)
                & (dist <= tol_min)
                & (cc != np.arange(n_rows))
            )
            better = good & (dist < bestd)
            best[better] = cc[better]
            bestd[better] = dist[better]

        m = best >= 0
        y = np.full(n_rows, np.nan, dtype=np.float32)
        y[m] = df["arrival_delay"].iloc[best[m]].to_numpy(dtype=np.float32)
        out[f"target_delay_{h}h"] = y

    return out


def build_dataset_split(
    start: str,
    end: str,
    scenario: str = "medium",
    stride: int = 4,
    tt: Optional[object] = None,
    nodes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load observations in [start, end], compute targets for 1-6h horizons, and extract features.

    stride: Sample cadence across consecutive station stops of a run. Real scheduled halts
    (where dwell > 0) are always retained to preserve critical junction and dwell events.
    """
    tt = tt or load_timetable()
    if nodes is None:
        nodes = build_station_tables(tt)[0]

    raw_df, _, _, bounds = load_window(start, end, scenario=scenario, tt=tt, nodes=nodes)
    arr_df = arrivals(raw_df, bounds)
    if len(arr_df) == 0:
        return pd.DataFrame()

    with_targets = compute_downstream_targets(arr_df, horizons=HORIZONS)
    with_features = extract_observation_features(with_targets, nodes_df=nodes)

    # Subsample if stride > 1, preserving all real halts and terminus stops
    if stride > 1:
        is_halt = with_features.get("halt", pd.Series(False, index=with_features.index))
        # Keep every stride-th stop OR any real halt
        sub_mask = (with_features["seq"] % stride == 0) | is_halt
        result = with_features[sub_mask].reset_index(drop=True)
    else:
        result = with_features.reset_index(drop=True)

    return result


def prepare_all_splits(
    output_dir: Path | str = DEFAULT_DATA_DIR,
    stride: int = 4,
    force_rebuild: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Generate and persist the official Train, Validation, Test, and Fold splits.

    Respects configs/splits.yaml windows:
    - train: 2024-01-01 to 2025-08-31
    - validation: 2025-09-01 to 2025-10-31
    - test: 2025-11-01 to 2025-12-31
    - fold1_fog: train 2024-01-01..2024-10-31, val 2024-11-01..2024-12-31
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / f"dataset_train_stride{stride}.parquet"
    val_file = out_path / f"dataset_val_stride{stride}.parquet"
    test_file = out_path / f"dataset_test_stride{stride}.parquet"
    fold1_val_file = out_path / f"dataset_fold1_val_stride{stride}.parquet"

    # Return cached parquets if already built
    if not force_rebuild and train_file.exists() and val_file.exists() and test_file.exists():
        return {
            "train": pd.read_parquet(train_file),
            "val": pd.read_parquet(val_file),
            "test": pd.read_parquet(test_file),
            "fold1_val": pd.read_parquet(fold1_val_file) if fold1_val_file.exists() else pd.DataFrame(),
        }

    tt = load_timetable()
    nodes = build_station_tables(tt)[0]

    # 1. Main training window
    tr_start, tr_end = splits.window("train")
    train_df = build_dataset_split(tr_start, tr_end, stride=stride, tt=tt, nodes=nodes)
    train_df.to_parquet(train_file, index=False)

    # 2. Validation window (Sep-Oct 2025)
    va_start, va_end = splits.window("validation")
    val_df = build_dataset_split(va_start, va_end, stride=stride, tt=tt, nodes=nodes)
    val_df.to_parquet(val_file, index=False)

    # 3. Untouched Test window (Nov-Dec 2025)
    te_start, te_end = splits.window("test")
    test_df = build_dataset_split(te_start, te_end, stride=stride, tt=tt, nodes=nodes)
    test_df.to_parquet(test_file, index=False)

    # 4. Fold 1 Validation window (Nov-Dec 2024 - contains fog)
    f1_va_start, f1_va_end = splits.fold("fold1_fog")["validate"]
    fold1_val_df = build_dataset_split(f1_va_start, f1_va_end, stride=stride, tt=tt, nodes=nodes)
    fold1_val_df.to_parquet(fold1_val_file, index=False)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "fold1_val": fold1_val_df,
    }


def get_horizon_subset(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Filter DataFrame to rows having a valid (non-null) target for horizon h."""
    target_col = f"target_delay_{horizon}h"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not present in DataFrame.")
    return df[df[target_col].notna()].copy()
