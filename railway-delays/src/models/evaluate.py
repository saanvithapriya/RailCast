"""Evaluation framework, metric calculations, and visualization for Phase 6 delay prediction.

Computes:
- MAE, RMSE, sample count (N), R-squared.
- Baseline-relative improvement: (MAE_baseline - MAE_model) / MAE_baseline * 100%.
- Segmentations:
  - Per horizon (1h, 2h, 3h, 4h, 5h, 6h)
  - Overall pooled evaluation
  - Fog versus non-fog periods
  - Train priority classes (Rajdhani, Superfast, Mail/Express, Passenger, etc.)
- Plots saved to reports/figures/:
  - Actual-versus-predicted plots
  - Error distribution histograms
  - Model comparison charts across horizons
- JSON summary saved to reports/phase6_metrics.json.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.dataset import HORIZONS, get_horizon_subset

DEFAULT_FIGURES_DIR = Path("reports/figures")
DEFAULT_METRICS_PATH = Path("reports/phase6_metrics.json")


def calc_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate MAE, RMSE, sample count, R2, and baseline improvement percentage."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    valid = ~(np.isnan(y_t) | np.isnan(y_p))
    if y_baseline is not None:
        y_b = np.asarray(y_baseline, dtype=np.float64)
        valid = valid & ~np.isnan(y_b)
    else:
        y_b = None

    y_t, y_p = y_t[valid], y_p[valid]
    n = len(y_t)
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "improvement_pct": np.nan}

    err = y_p - y_t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    improvement_pct = np.nan
    if y_b is not None:
        b_err = y_b[valid] - y_t
        b_mae = float(np.mean(np.abs(b_err)))
        if b_mae > 0:
            improvement_pct = float((b_mae - mae) / b_mae * 100.0)

    return {
        "n": int(n),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "improvement_pct": round(improvement_pct, 2) if not np.isnan(improvement_pct) else np.nan,
    }


def evaluate_predictions(
    df: pd.DataFrame,
    y_pred_col: str,
    y_true_col: str,
    y_baseline_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate predictions on a dataframe, with overall and weather/class slices."""
    valid_df = df[df[y_true_col].notna() & df[y_pred_col].notna()].copy()
    if len(valid_df) == 0:
        return {"overall": calc_metrics(np.array([]), np.array([]))}

    y_t = valid_df[y_true_col].to_numpy()
    y_p = valid_df[y_pred_col].to_numpy()
    y_b = valid_df[y_baseline_col].to_numpy() if y_baseline_col and y_baseline_col in valid_df.columns else None

    results = {
        "overall": calc_metrics(y_t, y_p, y_b),
        "weather": {},
        "train_type": {},
    }

    # Slice by weather (fog vs non-fog)
    if "is_fog" in valid_df.columns:
        for is_fog_val, label in ((1, "fog"), (0, "non_fog")):
            sub = valid_df[valid_df["is_fog"] == is_fog_val]
            if len(sub) > 0:
                y_tb = sub[y_baseline_col].to_numpy() if y_b is not None else None
                results["weather"][label] = calc_metrics(
                    sub[y_true_col].to_numpy(),
                    sub[y_pred_col].to_numpy(),
                    y_tb,
                )

    # Slice by train type
    if "type" in valid_df.columns:
        for t_type in valid_df["type"].dropna().unique():
            sub = valid_df[valid_df["type"] == t_type]
            if len(sub) >= 20:
                y_tb = sub[y_baseline_col].to_numpy() if y_b is not None else None
                results["train_type"][str(t_type)] = calc_metrics(
                    sub[y_true_col].to_numpy(),
                    sub[y_pred_col].to_numpy(),
                    y_tb,
                )

    return results


def evaluate_suite(
    df: pd.DataFrame,
    models: Dict[int, Any],
    pipeline: Any,
    persistence_model: Optional[Any] = None,
    historical_model: Optional[Any] = None,
    horizons: List[int] = HORIZONS,
) -> Dict[str, Any]:
    """Evaluate trained ML models and baselines across all six horizons."""
    summary = {
        "horizons": {},
        "macro_avg": {},
    }

    model_maes, pers_maes, hist_maes = [], [], []

    for h in horizons:
        sub = get_horizon_subset(df, horizon=h)
        if len(sub) == 0:
            continue

        target_col = f"target_delay_{h}h"
        y_true = sub[target_col].to_numpy(dtype=np.float32)

        # Baseline: Persistence
        pers_pred = sub["current_delay"].to_numpy(dtype=np.float32) if "current_delay" in sub else sub["arrival_delay"].to_numpy(dtype=np.float32)
        sub["pred_pers"] = pers_pred
        pers_metrics = calc_metrics(y_true, pers_pred)
        pers_maes.append(pers_metrics["mae"])

        # Baseline: Historical
        if historical_model is not None:
            hist_pred = historical_model.predict(sub, horizon=h)
            sub["pred_hist"] = hist_pred
            hist_metrics = calc_metrics(y_true, hist_pred, y_baseline=pers_pred)
            hist_maes.append(hist_metrics["mae"])
        else:
            hist_metrics = None

        # ML Model (e.g. LightGBM, RF)
        model = models.get(h)
        if model is not None:
            X = pipeline.transform(sub)
            ml_pred = model.predict(X)
            sub["pred_ml"] = ml_pred
            ml_metrics = evaluate_predictions(
                sub,
                y_pred_col="pred_ml",
                y_true_col=target_col,
                y_baseline_col="pred_pers",
            )
            model_maes.append(ml_metrics["overall"]["mae"])
        else:
            ml_metrics = None

        summary["horizons"][f"{h}h"] = {
            "persistence": pers_metrics,
            "historical": hist_metrics,
            "model": ml_metrics,
        }

    if model_maes:
        summary["macro_avg"]["model_mae"] = round(float(np.mean(model_maes)), 3)
    if pers_maes:
        summary["macro_avg"]["persistence_mae"] = round(float(np.mean(pers_maes)), 3)
    if hist_maes:
        summary["macro_avg"]["historical_mae"] = round(float(np.mean(hist_maes)), 3)
    if model_maes and pers_maes:
        overall_imp = (np.mean(pers_maes) - np.mean(model_maes)) / np.mean(pers_maes) * 100.0
        summary["macro_avg"]["overall_improvement_vs_persistence_pct"] = round(float(overall_imp), 2)

    return summary


# ----------------------------------------------------------------------------- Plotting Functions
def plot_model_comparison(
    metrics_summary: Dict[str, Any],
    out_path: Path | str = DEFAULT_FIGURES_DIR / "phase6_model_comparison.png",
):
    """Plot MAE and RMSE across all 6 horizons comparing Persistence, Historical, and ML models."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    horizons = [h for h in (1, 2, 3, 4, 5, 6) if f"{h}h" in metrics_summary["horizons"]]
    h_labels = [f"{h}h" for h in horizons]

    pers_mae = [metrics_summary["horizons"][hl]["persistence"]["mae"] for hl in h_labels]
    hist_mae = [
        metrics_summary["horizons"][hl]["historical"]["mae"]
        if metrics_summary["horizons"][hl]["historical"]
        else np.nan
        for hl in h_labels
    ]
    model_mae = [
        metrics_summary["horizons"][hl]["model"]["overall"]["mae"]
        if metrics_summary["horizons"][hl]["model"]
        else np.nan
        for hl in h_labels
    ]

    pers_rmse = [metrics_summary["horizons"][hl]["persistence"]["rmse"] for hl in h_labels]
    model_rmse = [
        metrics_summary["horizons"][hl]["model"]["overall"]["rmse"]
        if metrics_summary["horizons"][hl]["model"]
        else np.nan
        for hl in h_labels
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # MAE plot
    axes[0].plot(horizons, pers_mae, marker="o", color="#d95f02", lw=2, label="Persistence Baseline")
    if any(~np.isnan(hist_mae)):
        axes[0].plot(horizons, hist_mae, marker="s", color="#7570b3", lw=1.5, ls="--", label="Historical Baseline")
    axes[0].plot(horizons, model_mae, marker="^", color="#1b9e77", lw=2.5, label="LightGBM Regressor")
    axes[0].set_title("Downstream Delay Prediction: MAE across Horizons [SIMULATED]", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Forecast Horizon (Hours)", fontsize=10)
    axes[0].set_ylabel("Mean Absolute Error (Minutes)", fontsize=10)
    axes[0].set_xticks(horizons)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=True)

    # RMSE plot
    axes[1].plot(horizons, pers_rmse, marker="o", color="#d95f02", lw=2, label="Persistence Baseline")
    axes[1].plot(horizons, model_rmse, marker="^", color="#1b9e77", lw=2.5, label="LightGBM Regressor")
    axes[1].set_title("Downstream Delay Prediction: RMSE across Horizons [SIMULATED]", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Forecast Horizon (Hours)", fontsize=10)
    axes[1].set_ylabel("Root Mean Squared Error (Minutes)", fontsize=10)
    axes[1].set_xticks(horizons)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    out_path: Path | str,
    max_val: float = 180.0,
):
    """Hexbin / scatter of actual versus predicted delay with 45-degree identity line."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    hb = ax.hexbin(y_true, y_pred, gridsize=50, cmap="viridis", mincnt=1, bins="log")
    cb = fig.colorbar(hb, ax=ax, label="Log10 Count")

    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect Forecast (Identity)")
    ax.set_xlim(-5, max_val)
    ax.set_ylim(-5, max_val)
    ax.set_title(f"Actual vs Predicted Delay: Horizon {horizon}h [SIMULATED]", fontsize=11, fontweight="bold")
    ax.set_xlabel("Actual Downstream Delay (Minutes)", fontsize=10)
    ax.set_ylabel("Predicted Downstream Delay (Minutes)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_error_distributions(
    errors_dict: Dict[int, np.ndarray],
    out_path: Path | str = DEFAULT_FIGURES_DIR / "phase6_error_distributions.png",
):
    """Histograms of prediction errors across horizons."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    horizons = sorted(list(errors_dict.keys()))
    n_h = len(horizons)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150, sharex=True, sharey=True)
    axes = axes.flatten()

    for i, h in enumerate(horizons):
        err = errors_dict[h]
        err_clipped = np.clip(err, -60, 60)
        axes[i].hist(err_clipped, bins=50, color="#1b9e77", alpha=0.7, edgecolor="black", lw=0.5)
        axes[i].axvline(0, color="red", linestyle="--", lw=1.5)
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        axes[i].set_title(f"Horizon {h}h (MAE={mae:.1f}m, RMSE={rmse:.1f}m)", fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Prediction Error: (Pred - Actual) min", fontsize=9)
        axes[i].set_ylabel("Count", fontsize=9)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Prediction Error Distributions by Horizon [SIMULATED]", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def save_metrics_json(
    metrics_data: Dict[str, Any],
    out_path: Path | str = DEFAULT_METRICS_PATH,
):
    """Save machine-readable evaluation metrics to JSON."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
