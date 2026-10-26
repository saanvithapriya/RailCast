"""Time split and simulated-data sources, read from configs/splits.yaml (the only place they are defined).

SIMULATED data. Windows are inclusive date pairs (start, end) given as 'YYYY-MM-DD' strings.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "splits.yaml"


@dataclass(frozen=True)
class Source:
    """One simulation run and the calendar range it is allowed to supply."""
    dir: str
    start: str
    end: str
    seed: int = 0


def _cfg():
    return yaml.safe_load(CONFIG_PATH.read_text())


def _pair(v):
    return str(v[0]), str(v[1])


def windows():
    """{'train': (start, end), 'validation': ..., 'test': ..., 'analysis_all_2025': ...}"""
    return {k: _pair(v) for k, v in _cfg()["splits"].items()}


def window(name):
    return windows()[name]


def sources(scenario="medium"):
    return [Source(str(s["dir"]), str(s["start"]), str(s["end"]), int(s.get("seed", 0))) for s in _cfg()["scenarios"][scenario]["sources"]]


def window_tag(name):
    """File-name tag for delay tables built on a window, derived from the config."""
    a, b = window(name)
    if name == "train":
        return f"SIMULATED_train_{a}_{b}"
    if name.startswith("analysis"):
        return f"SIMULATED_ANALYSIS_ONLY_{name.split('analysis_')[-1]}"
    return f"SIMULATED_{name}_{a}_{b}"


# ----------------------------------------------------------------------------- rolling-origin folds
def folds():
    """{'fold1_fog': {'train': (a, b), 'validate': (a, b)}, 'fold2_clear': {...}} from configs/splits.yaml."""
    return {k: {"train": _pair(v["train"]), "validate": _pair(v["validate"])} for k, v in _cfg()["folds"].items()}


def fold(name):
    return folds()[name]


def fold_names():
    return list(folds())


def fold_tag(name, part="train"):
    """File-name tag for delay tables built on one fold's training or validation range."""
    a, b = fold(name)[part]
    return f"SIMULATED_{name}_{part}_{a}_{b}"


def eda_windows():
    """{'decision': (a, b), 'descriptive': (a, b)}: Phase 4 uses 'decision' for anything that informs modelling."""
    return {k: _pair(v) for k, v in _cfg()["eda"].items()}
