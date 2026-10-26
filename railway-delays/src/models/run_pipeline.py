"""End-to-end execution script for Phase 6 downstream delay prediction pipeline.

Workflow:
1. Dataset preparation: Build or load official splits from configs/splits.yaml.
2. Baselines: Fit Persistence and Historical baselines on training data.
3. Model training: Train LightGBM regressors for horizons 1-6h; train Random Forest and GNN.
4. Validation & Model Selection: Evaluate on Validation split (Sep-Oct 2025) and Fold 1 Fog split.
5. Untouched Test Evaluation: Evaluate final selected models on held-out Test split (Nov-Dec 2025).
6. Artifacts & Reporting: Save plots to reports/figures/, metrics to reports/phase6_metrics.json,
   and model bundle to data/processed/models/.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.models.artifacts import DEFAULT_MODEL_DIR, save_model_bundle
from src.models.baseline import HistoricalPredictor, PersistencePredictor
from src.models.dataset import HORIZONS, get_horizon_subset, prepare_all_splits
from src.models.evaluate import (
    calc_metrics,
    evaluate_suite,
    plot_actual_vs_predicted,
    plot_error_distributions,
    plot_model_comparison,
    save_metrics_json,
)
from src.models.features import FeaturePipeline
from src.models.gnn import GNNRegressor
from src.models.train import train_full_pipeline

FIGURES_DIR = Path("reports/figures")
METRICS_PATH = Path("reports/phase6_metrics.json")


def run_phase6_pipeline(
    stride: int = 4,
    force_rebuild: bool = False,
    run_gnn: bool = True,
    run_rf: bool = True,
    max_train_samples: int = 250000,
) -> Dict[str, Any]:
    """Execute complete Phase 6 training and evaluation pipeline."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RAILCAST PHASE 6: END-TO-END DOWNSTREAM DELAY PREDICTION PIPELINE")
    print("=" * 80)

    # 1. Dataset loading / generation
    t0 = time.time()
    print(f"\n[1/6] Preparing forecasting dataset (stride={stride})...")
    splits_dict = prepare_all_splits(stride=stride, force_rebuild=force_rebuild)
    train_df = splits_dict["train"]
    val_df = splits_dict["val"]
    test_df = splits_dict["test"]
    fold1_val_df = splits_dict.get("fold1_val", pd.DataFrame())

    print(f"  Train set:      {len(train_df):,} examples (2024-01-01 to 2025-08-31)")
    print(f"  Validation set: {len(val_df):,} examples (2025-09-01 to 2025-10-31)")
    print(f"  Test set:       {len(test_df):,} examples (2025-11-01 to 2025-12-31, UNTOUCHED)")
    if len(fold1_val_df) > 0:
        print(f"  Fold 1 Val set: {len(fold1_val_df):,} examples (2024-11-01 to 2024-12-31, 12 fog days)")
    print(f"  Dataset preparation completed in {time.time() - t0:.1f}s.")

    # 2. Fit Baselines on Training Split
    t0 = time.time()
    print("\n[2/6] Fitting Baselines strictly on Training data...")
    persistence = PersistencePredictor().fit(train_df)
    historical = HistoricalPredictor().fit(train_df)
    print(f"  Baselines fitted in {time.time() - t0:.1f}s.")

    # 3. Train Classical ML Models (LightGBM Primary)
    t0 = time.time()
    print("\n[3/6] Training Primary LightGBM Regressors across all 6 horizons...")
    lgbm_models, pipeline = train_full_pipeline(
        train_df,
        val_df=val_df,
        model_type="lightgbm",
        horizons=HORIZONS,
        bundle_dir=DEFAULT_MODEL_DIR,
        max_train_samples=max_train_samples,
        random_state=42,
    )
    print(f"  LightGBM models trained and saved to {DEFAULT_MODEL_DIR} in {time.time() - t0:.1f}s.")

    # Optional: Random Forest Regressors for comparison
    rf_models = None
    if run_rf:
        t0 = time.time()
        print("\n[3b/6] Training Random Forest Regressors (subset sample for feasibility)...")
        rf_models, _ = train_full_pipeline(
            train_df,
            model_type="rf",
            horizons=HORIZONS,
            max_train_samples=40000,
            n_estimators=50,
            max_depth=12,
            random_state=42,
        )
        print(f"  Random Forest models trained in {time.time() - t0:.1f}s.")

    # Optional: Graph Neural Network Experiment
    gnn_models = {}
    if run_gnn:
        t0 = time.time()
        print("\n[3c/6] Training Spatio-Temporal Graph Neural Network (PyG) for key horizons...")
        for h in [1, 3, 6]:
            print(f"  Training GNN for horizon {h}h...")
            sub_tr = get_horizon_subset(train_df, horizon=h)
            if len(sub_tr) > 50000:
                sub_tr = sub_tr.sample(n=50000, random_state=42 + h)
            X_tr = pipeline.transform(sub_tr)
            y_tr = sub_tr[f"target_delay_{h}h"].to_numpy(dtype=np.float32)

            gnn = GNNRegressor(horizon=h, epochs=8, batch_size=512, random_state=42)
            gnn.fit(X_tr, y_tr)
            gnn_models[h] = gnn
        print(f"  GNN models trained in {time.time() - t0:.1f}s.")

    # 4. Validation Evaluation & Model Selection
    print("\n[4/6] Evaluating on Validation set (Sep-Oct 2025)...")
    val_lgbm = evaluate_suite(val_df, lgbm_models, pipeline, persistence, historical)

    val_rf = evaluate_suite(val_df, rf_models, pipeline, persistence, historical) if rf_models else None
    val_gnn = evaluate_suite(val_df, gnn_models, pipeline, persistence, historical) if gnn_models else None

    # Evaluate Fold 1 Validation (Fog robustness check)
    fold1_lgbm = None
    if len(fold1_val_df) > 0:
        print("  Evaluating on Fold 1 Validation (Fog Season Nov-Dec 2024)...")
        fold1_lgbm = evaluate_suite(fold1_val_df, lgbm_models, pipeline, persistence, historical)

    # 5. Untouched Test Set Evaluation
    print("\n[5/6] Evaluating final selected LightGBM model on UNTOUCHED Test set (Nov-Dec 2025)...")
    test_lgbm = evaluate_suite(test_df, lgbm_models, pipeline, persistence, historical)
    test_rf = evaluate_suite(test_df, rf_models, pipeline, persistence, historical) if rf_models else None

    # Print summary table
    print("\n" + "=" * 80)
    print("VALIDATION SET RESULTS (Sep-Oct 2025) [SIMULATED]")
    print(f"{'Horizon':<10} | {'Persistence MAE':<16} | {'Historical MAE':<16} | {'LightGBM MAE':<14} | {'Improvement vs Pers':<20}")
    print("-" * 80)
    for h in HORIZONS:
        hl = f"{h}h"
        p_mae = val_lgbm["horizons"][hl]["persistence"]["mae"]
        h_mae = val_lgbm["horizons"][hl]["historical"]["mae"]
        m_mae = val_lgbm["horizons"][hl]["model"]["overall"]["mae"]
        imp = val_lgbm["horizons"][hl]["model"]["overall"]["improvement_pct"]
        print(f"{hl:<10} | {p_mae:<16.2f} | {h_mae:<16.2f} | {m_mae:<14.2f} | {imp:>+18.2f}%")

    print("\n" + "=" * 80)
    print("UNTOUCHED TEST SET RESULTS (Nov-Dec 2025, 10 Fog Days) [SIMULATED]")
    print(f"{'Horizon':<10} | {'Persistence MAE':<16} | {'Historical MAE':<16} | {'LightGBM MAE':<14} | {'Improvement vs Pers':<20}")
    print("-" * 80)
    for h in HORIZONS:
        hl = f"{h}h"
        p_mae = test_lgbm["horizons"][hl]["persistence"]["mae"]
        h_mae = test_lgbm["horizons"][hl]["historical"]["mae"]
        m_mae = test_lgbm["horizons"][hl]["model"]["overall"]["mae"]
        imp = test_lgbm["horizons"][hl]["model"]["overall"]["improvement_pct"]
        print(f"{hl:<10} | {p_mae:<16.2f} | {h_mae:<16.2f} | {m_mae:<14.2f} | {imp:>+18.2f}%")
    print("=" * 80)

    # 6. Generate Figures and Save Reports
    print("\n[6/6] Generating evaluation plots and metrics JSON...")
    plot_model_comparison(test_lgbm, out_path=FIGURES_DIR / "phase6_model_comparison.png")

    # Hexbin actual vs predicted for 1h and 6h
    errors_dict = {}
    for h in HORIZONS:
        sub_te = get_horizon_subset(test_df, horizon=h)
        X_te = pipeline.transform(sub_te)
        y_true = sub_te[f"target_delay_{h}h"].to_numpy()
        y_pred = lgbm_models[h].predict(X_te)
        errors_dict[h] = y_pred - y_true

        if h in (1, 6):
            plot_actual_vs_predicted(
                y_true,
                y_pred,
                horizon=h,
                out_path=FIGURES_DIR / f"phase6_actual_vs_predicted_{h}h.png",
            )

    plot_error_distributions(errors_dict, out_path=FIGURES_DIR / "phase6_error_distributions.png")

    # Save metrics JSON
    metrics_summary = {
        "validation_lgbm": val_lgbm,
        "test_lgbm": test_lgbm,
        "fold1_val_lgbm": fold1_lgbm,
        "validation_rf": val_rf,
        "test_rf": test_rf,
        "validation_gnn": val_gnn,
    }
    save_metrics_json(metrics_summary, out_path=METRICS_PATH)
    print(f"  Evaluation metrics saved to {METRICS_PATH}")
    print(f"  Plots saved to {FIGURES_DIR}")
    print("\nPipeline execution complete!")

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RailCast Phase 6 prediction pipeline.")
    parser.add_argument("--stride", type=int, default=4, help="Sampling stride across run stops (default: 4)")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild of dataset parquets")
    parser.add_argument("--skip-gnn", action="store_true", help="Skip GNN experiment")
    parser.add_argument("--skip-rf", action="store_true", help="Skip Random Forest baseline")
    parser.add_argument("--max-train", type=int, default=250000, help="Max train samples per horizon")

    args = parser.parse_args()
    run_phase6_pipeline(
        stride=args.stride,
        force_rebuild=args.force_rebuild,
        run_gnn=not args.skip_gnn,
        run_rf=not args.skip_rf,
        max_train_samples=args.max_train,
    )
