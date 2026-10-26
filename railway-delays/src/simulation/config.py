"""Config loading: defaults from config.yaml, optional overrides (used by sensitivity runs)."""
import copy
from pathlib import Path
from types import SimpleNamespace

import yaml

DEFAULT_PATH = Path(__file__).with_name("config.yaml")


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None, overrides=None):
    """Return the config as a nested dict, with `overrides` (nested dict) merged on top."""
    cfg = yaml.safe_load(Path(path or DEFAULT_PATH).read_text())
    return _merge(cfg, copy.deepcopy(overrides or {}))


def ns(d):
    """Nested dict -> attribute access (read-only convenience)."""
    return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()})
