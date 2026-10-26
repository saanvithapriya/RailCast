import numpy as np
import pandas as pd
import pytest

from src.analysis import robustness as rb


def crit(v):
    v = np.asarray(v, float)
    return pd.DataFrame({"sec": range(len(v)), "origin_freq": v, "size_triggered": v, "primary_minutes": v, "pagerank_station": v, "edge_betweenness": v})


def test_identical_and_reversed_rankings():
    r = rb.ranking_stability({"a": crit(range(20)), "b": crit(range(20)), "c": crit(range(20, 0, -1))}, top=5)
    ab = r[r.pair == "a vs b"]; ac = r[r.pair == "a vs c"]
    assert (ab.spearman == 1).all() and (ab.top_overlap == 5).all() and (ac.spearman == -1).all() and (ac.top_overlap == 0).all()


def test_size_distribution():
    c = pd.DataFrame({"size_runs": [1, 1, 2, 4, 30], "depth": 1, "extent_sections": 2})
    d = rb.size_distribution(c)
    assert d["n_cascades"] == 5 and d["share>=2"] == 0.6 and d["max"] == 30


def test_contagion_stability():
    t = pd.DataFrame({"sec": [0, 0, 1, 1, 2, 2], "dir": [1, -1] * 3, "rho": [.1, .2, .3, .4, .5, .6], "A_all": 1.0, "A_primary": 1.0, "beta": 1.0, "c": 1.0})
    t["A_all"] = np.arange(6); t["A_primary"] = np.arange(6); t["beta"] = np.arange(6); t["c"] = np.arange(6)
    r = rb.contagion_stability({"x": t, "y": t.assign(rho=t.rho * 2)})
    assert r[r.measure == "rho"].spearman.iloc[0] == pytest.approx(1.0) and r[r.measure == "rho"].median_y.iloc[0] == pytest.approx(2 * r[r.measure == "rho"].median_x.iloc[0])
