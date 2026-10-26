"""Item 6: station communities from delay correlation (SIMULATED data, observable inputs only).

Each node gets an hourly series of the mean arrival delay of the trains arriving there (both directions pooled, gaps of
up to 3 hours interpolated, the node's hour-of-day mean removed so that the shared daily pattern does not create
correlation). Nodes are clustered by 1 - correlation with average-linkage hierarchical clustering. The number of
clusters is chosen by the silhouette score on the DECISION window, without looking at the physical segments, and is
then frozen (configs/mining_frozen.yaml) and applied unchanged to later windows. A modularity (Louvain) clustering of the
thresholded correlation graph is a cross-check. Clusters are compared with the physical segments NDLS-TDL, TDL-ETW,
ETW-CNB, CNB-PRYG, PRYG-MGS by the adjusted Rand index, normalised mutual information and contiguity along the line.
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

FROZEN = Path(__file__).resolve().parent.parent.parent / "configs" / "mining_frozen.yaml"
ANCHORS = ["NDLS", "TDL", "ETW", "CNB", "PRYG", "MGS"]


def node_series(mean, cnt, deseason=True, limit=3):
    """(nodes x hours) pooled hourly mean arrival delay, gaps interpolated, hour-of-day mean removed."""
    num = np.where(cnt > 0, mean * cnt, 0.0).sum(axis=1)
    den = cnt.sum(axis=1)
    pooled = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    out = np.vstack([pd.Series(r).interpolate(limit=limit, limit_area="inside").to_numpy() for r in pooled])
    if deseason:
        hod = np.arange(out.shape[1]) % 24
        for h in range(24):
            sel = hod == h
            out[:, sel] = out[:, sel] - np.nanmean(out[:, sel], axis=1, keepdims=True)
    return out


def correlation_matrix(series):
    c = np.array(pd.DataFrame(series.T).corr().to_numpy(), copy=True)
    np.fill_diagonal(c, 1.0)
    return np.nan_to_num(c, nan=0.0)


def cluster(corr, k, method="average"):
    d = np.clip(1.0 - corr, 0.0, None)
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method=method)
    return fcluster(z, k, criterion="maxclust") - 1


def choose_k(corr, ks=range(2, 11), method="average"):
    """k with the best silhouette score (precomputed 1 - correlation distances)."""
    d = np.clip(1.0 - corr, 0.0, None)
    np.fill_diagonal(d, 0.0)
    scores = {k: float(silhouette_score(d, cluster(corr, k, method), metric="precomputed")) for k in ks}
    return max(scores, key=scores.get), scores


def louvain_labels(corr, tau=0.5, seed=0):
    """Louvain communities of the graph whose edges are the correlations at least tau (weight = correlation)."""
    n = len(corr)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    ii, jj = np.where(np.triu(corr >= tau, 1))
    g.add_weighted_edges_from((int(i), int(j), float(corr[i, j])) for i, j in zip(ii, jj))
    comms = nx.community.louvain_communities(g, weight="weight", seed=seed)
    lab = np.zeros(n, dtype=int)
    for c, nodes in enumerate(comms):
        lab[list(nodes)] = c
    return lab


def physical_segments(node_names, anchors=ANCHORS):
    """Segment index of every node along the chain given the anchor station names."""
    idx = [node_names.index(a) for a in anchors]
    return np.clip(np.searchsorted(idx, np.arange(len(node_names)), side="right") - 1, 0, len(idx) - 2)


def contiguity(labels):
    """Number of contiguous blocks along the chain and the share of adjacent node pairs in the same cluster."""
    lab = np.asarray(labels)
    blocks = 1 + int((lab[1:] != lab[:-1]).sum())
    return blocks, float((lab[1:] == lab[:-1]).mean())


def compare_with_segments(labels, segments):
    blocks, same = contiguity(labels)
    return {"clusters": int(len(set(labels))), "ARI": float(adjusted_rand_score(segments, labels)),
            "NMI": float(normalized_mutual_info_score(segments, labels)), "blocks_along_chain": blocks, "adjacent_same_cluster": same}


def load_frozen_communities():
    c = yaml.safe_load(FROZEN.read_text())
    return c.get("communities")


def calibrate_communities(hourly_npz, ks=range(2, 11), method="average"):
    """Choose k on the decision window and write it into configs/mining_frozen.yaml (once)."""
    h = np.load(hourly_npz)
    corr = correlation_matrix(node_series(h["mean"], h["count"]))
    k, scores = choose_k(corr, ks, method)
    cfg = yaml.safe_load(FROZEN.read_text())
    cfg["communities"] = {"k": int(k), "linkage": method, "deseason": True, "louvain_tau": 0.5, "silhouette": {int(a): round(b, 4) for a, b in scores.items()}}
    FROZEN.write_text("# FROZEN mining parameters: calibrated on 2024-01-01..2024-10-31 only (SIMULATED data); never re-estimated.\n"
                      + yaml.safe_dump(cfg, sort_keys=False))
    return k, scores


def communities_for_window(hourly_npz, node_names, frozen=None, k=None):
    """Apply the frozen method to a window's hourly arrays. Returns a dict with labels and comparison numbers.

    k overrides the frozen cluster count (used only for the extra comparison with k equal to the number of segments)."""
    fz = frozen or load_frozen_communities()
    h = np.load(hourly_npz)
    corr = correlation_matrix(node_series(h["mean"], h["count"], fz["deseason"]))
    lab = cluster(corr, k or fz["k"], fz["linkage"])
    lv = louvain_labels(corr, fz["louvain_tau"])
    seg = physical_segments(list(node_names))
    return {"corr": corr, "labels": lab, "louvain": lv, "segments": seg,
            "hierarchical": compare_with_segments(lab, seg), "louvain_cmp": compare_with_segments(lv, seg)}
