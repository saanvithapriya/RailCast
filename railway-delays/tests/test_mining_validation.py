"""Validation helpers on tiny hand-made inferred / true tables, and the guards that keep the truth away from the miners."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import validation as va


def truth_rows():
    # follower A waited behind leader L1 (cause 4); B was held for a priority train H (cause 5); C had an incident (cause 1)
    return pd.DataFrame({
        "train": ["A", "B", "C", "D"], "day": [1, 1, 1, 1], "to_node": [5, 6, 7, 8], "sec": [4, 5, 6, 7],
        "cause_code": [4, 5, 1, 4], "cause_train": ["L1", "H", "", "L9"], "new_minutes": [6.0, 4.0, 20.0, 3.0]})


def test_link_precision_and_recall():
    inferred = pd.DataFrame({"train": ["A", "B", "D", "E"], "day": 1, "to_node": [5, 6, 8, 9], "leader": ["L1", "X", "WRONG", "L2"]})
    r = va.link_precision_recall(inferred, truth_rows())
    # A is right; B is wrong (it was held for H, the inferred leader X is not the cause); D wrong leader; E has no truth
    assert r == {"inferred": 4, "true": 3, "correct": 1, "precision": 0.25, "recall": pytest.approx(1 / 3)}
    rc = va.recall_by_cause(inferred, truth_rows()).set_index("cause_code")
    assert rc.loc[4, "recovered"] == 1 and rc.loc[5, "recovered"] == 0 and rc.loc[1, "recall"] == 0     # yields and incidents are not links


def test_false_positives_are_classified():
    inferred = pd.DataFrame({"train": ["B", "C", "E"], "day": 1, "to_node": [6, 7, 9], "leader": ["X", "L", "L"]})
    s = va.false_positive_breakdown(inferred, truth_rows())
    assert s["wrong leader (another train was the cause)"] == 1 and s["caused by something other than a train"] == 1 and s["no label at that row"] == 1


def test_what_caused_the_inferred_primary_events():
    prim = pd.DataFrame({"train": ["C", "A", "Z"], "day": 1, "sec": [6, 4, 99]})
    counts, j = va.origin_cause_breakdown(prim, truth_rows())
    assert counts["incident"] == 1 and counts["section wait (leader)"] == 1 and counts["no label (below 0.5 min)"] == 1


def test_origin_matching_uses_train_section_and_time():
    casc = pd.DataFrame({"origin_train": ["C", "C", "Q"], "origin_sec": [6, 6, 1], "origin_time": [100.0, 900.0, 5.0]})
    ev = pd.DataFrame({"event_id": [11], "train": ["C"], "section": [6], "start_min": [101.0]})
    m = va.match_origins(casc, ev, tol=5)
    assert len(m) == 1 and m.iloc[0].event_id == 11 and m.iloc[0].origin_time == 100.0


def test_true_other_train_share_per_section():
    t = pd.DataFrame({"sec": [1, 1, 1], "dir": [1, 1, 1], "cause_code": [4, 1, 5], "new_minutes": [6.0, 20.0, 4.0]})
    s = va.other_train_share(t)
    assert s.true_other_share.iloc[0] == pytest.approx(10.0 / 30.0)
    assert va.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)


def test_loaders_refuse_without_the_validation_flag():
    with pytest.raises(PermissionError):
        va.load_truth_links("data/processed/sim/medium", "2025-01-01", "2025-01-31", {}, {})


def test_no_mining_module_imports_the_validation_or_ground_truth_code():
    forbidden = ("validation", "groundtruth", "load_groundtruth")
    for f in Path("src/analysis").glob("*.py"):
        if f.name in ("validation.py",):
            continue
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                mods = [n.module or ""] + [a.name for a in n.names]
            elif isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            else:
                continue
            assert not any(w in m for m in mods for w in forbidden), (f.name, mods)
