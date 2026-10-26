"""Model artifact persistence and metadata management for Phase 6 delay prediction.

Handles serialization and loading of trained models, feature pipelines, target definitions,
and reproducibility metadata (package versions, hyperparameters, and split windows).
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_DIR = Path("data/processed/models")


def get_library_versions() -> Dict[str, str]:
    """Capture runtime library versions for reproducibility."""
    versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }
    for pkg_name in ("sklearn", "lightgbm", "xgboost", "torch", "torch_geometric"):
        try:
            mod = __import__(pkg_name)
            versions[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg_name] = "not_installed"
    return versions


def save_model_bundle(
    bundle_dir: Path | str,
    models: Dict[int, Any],
    preprocessor: Any,
    metadata: Dict[str, Any],
) -> Path:
    """Save trained models, feature preprocessor, and metadata JSON to bundle_dir."""
    out_dir = Path(bundle_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save horizon models
    for h, model in models.items():
        model_path = out_dir / f"model_h{h}.joblib"
        joblib.dump(model, model_path)

    # Save preprocessor
    prep_path = out_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, prep_path)

    # Enrich and save metadata
    meta = dict(metadata)
    meta["horizons"] = sorted(list(models.keys()))
    meta["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta["library_versions"] = get_library_versions()
    meta["target_definition"] = (
        "Downstream arrival delay (minutes) of the train at the station reached closest to t_now + h hours"
    )

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_dir


def load_model_bundle(bundle_dir: Path | str) -> Dict[str, Any]:
    """Load model bundle: models dictionary {h: model}, preprocessor, and metadata."""
    in_dir = Path(bundle_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Model bundle directory not found: {in_dir}")

    meta_file = in_dir / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file missing in bundle: {meta_file}")

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    prep_file = in_dir / "preprocessor.joblib"
    if not prep_file.exists():
        raise FileNotFoundError(f"Preprocessor missing in bundle: {prep_file}")
    preprocessor = joblib.load(prep_file)

    models = {}
    for h in metadata.get("horizons", [1, 2, 3, 4, 5, 6]):
        m_path = in_dir / f"model_h{h}.joblib"
        if m_path.exists():
            models[int(h)] = joblib.load(m_path)
        else:
            raise FileNotFoundError(f"Model for horizon {h} missing: {m_path}")

    return {
        "models": models,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "bundle_dir": in_dir,
    }
