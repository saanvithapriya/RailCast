"""Item 6 (station communities) with synthetic blocks of correlated nodes."""
import numpy as np
import pytest

from src.analysis import communities as cm


def block_series(n_per=10, hours=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=hours), rng.normal(size=hours)
    rows = [a + 0.4 * rng.normal(size=hours) for _ in range(n_per)] + [b + 0.4 * rng.normal(size=hours) for _ in range(n_per)]
    return np.array(rows)


def test_k_is_found_without_labels_and_clusters_match_the_blocks():
    corr = cm.correlation_matrix(block_series())
    k, scores = cm.choose_k(corr)
    assert k == 2 and scores[2] == max(scores.values())
    lab = cm.cluster(corr, 2)
    assert len(set(lab[:10])) == 1 and len(set(lab[10:])) == 1 and lab[0] != lab[-1]
    truth = np.r_[np.zeros(10, int), np.ones(10, int)]
    assert cm.compare_with_segments(lab, truth)["ARI"] == pytest.approx(1.0)


def test_louvain_agrees_with_the_blocks():
    corr = cm.correlation_matrix(block_series())
    lv = cm.louvain_labels(corr, tau=0.5)
    assert len(set(lv[:10])) == 1 and len(set(lv[10:])) == 1 and lv[0] != lv[-1]


def test_contiguity_and_segments():
    assert cm.contiguity([0, 0, 1, 1, 0]) == (3, 0.5)
    names = ["NDLS", "a", "TDL", "b", "ETW", "c", "CNB", "d", "PRYG", "e", "MGS"]
    assert cm.physical_segments(names).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4]


def test_node_series_removes_the_hour_of_day_pattern_and_fills_short_gaps():
    hours = 24 * 30
    base = np.tile(np.arange(24, dtype="float32"), 30)
    mean = np.stack([base, base])[:, None, :].repeat(2, axis=1)           # two nodes, two directions, a pure daily pattern
    cnt = np.ones_like(mean, dtype="int16")
    s = cm.node_series(mean, cnt)
    assert np.nanmax(np.abs(s)) < 1e-4                                    # nothing is left once the daily mean is removed
    mean2, cnt2 = mean.copy(), cnt.copy()
    mean2[0, :, 100:102], cnt2[0, :, 100:102] = np.nan, 0
    s2 = cm.node_series(mean2, cnt2, deseason=False, limit=3)
    assert not np.isnan(s2[0, 100:102]).any()                              # a two-hour gap is interpolated


def test_frozen_community_parameters_are_stored_on_the_decision_window(tmp_path, monkeypatch):
    import yaml
    f = tmp_path / "frozen.yaml"
    f.write_text("G: 6\n")
    monkeypatch.setattr(cm, "FROZEN", f)
    np.savez(tmp_path / "h.npz", mean=block_series()[:, None, :].repeat(2, axis=1).astype("float32"), count=np.ones((20, 2, 2000), dtype="int16"))
    k, _ = cm.calibrate_communities(tmp_path / "h.npz")
    assert yaml.safe_load(f.read_text())["communities"]["k"] == k == 2
