"""Item 4: critical sections and trains (SIMULATED data, observable inputs only).

Section rankings (107 sections, both directions pooled):
  cascade-based   origin_freq (primary-event cascades starting in the section), primary_minutes (delay minutes added by
                  primary events there), size_triggered (runs in the cascades they start)
  station graph   PageRank and edge betweenness on the directed station graph weighted by the contagion coefficient of
                  each section and direction (pass-on ratio A_all; the linear coefficient is ~0 everywhere)
  train graph     summed PageRank of the trains scheduled through the section (train-dependency graph, timetable only)
Train rankings: origin_cascades, leader_links (times the train was the leader a follower was linked to),
caused_minutes (follower delay minutes linked to it), and PageRank / betweenness on the train-dependency graph.
Betweenness on a chain graph is topological (shortest paths are unique), so it mostly ranks the middle of the line.
"""
import networkx as nx
import numpy as np
import pandas as pd


def station_digraph(nodes, edges, contagion, weight_col="A_all", eps=1e-3):
    """Directed station graph; edge weight = contagion coefficient per (section, direction), distance = 1 / weight."""
    w = contagion.set_index(["sec", "dir"])[weight_col]
    g = nx.DiGraph()
    g.add_nodes_from(int(n) for n in nodes.node)
    for r in edges.itertuples():
        x = float(w.get((r.section, r.direction), np.nan))
        x = eps if not np.isfinite(x) else max(x, eps)
        g.add_edge(int(r.from_node), int(r.to_node), sec=int(r.section), dir=int(r.direction), w=x, dist=1.0 / x)
    return g


def train_digraph(train_edges, train_nodes):
    """Directed train-dependency graph; undirected section-sharing edges become two arcs; weight = count."""
    g = nx.DiGraph()
    g.add_nodes_from(train_nodes.train)
    for r in train_edges.itertuples():
        pairs = [(r.source, r.target)] + ([(r.target, r.source)] if r.type == "section_sharing" else [])
        for a, b in pairs:
            w = g[a][b]["w"] + r._asdict()["count"] if g.has_edge(a, b) else float(r._asdict()["count"])
            g.add_edge(a, b, w=w, dist=1.0 / max(w, 1e-9))
    return g


def critical_sections(cascades, primary_events, nodes, sg, tg, trains_through_section, n_sections):
    """One row per section with cascade metrics, graph centralities and the rank of each (1 = most critical)."""
    prim = cascades[cascades.origin_type == "primary"]
    d = pd.DataFrame({"sec": np.arange(n_sections)}).set_index("sec")
    d["origin_freq"] = prim.groupby("origin_sec").size().reindex(d.index).fillna(0)
    d["size_triggered"] = prim.groupby("origin_sec").size_runs.sum().reindex(d.index).fillna(0)
    d["primary_minutes"] = primary_events.groupby("sec").added.sum().reindex(d.index).fillna(0)
    d["external_origin_freq"] = cascades[cascades.origin_type == "external_entry"].groupby("origin_sec").size().reindex(d.index).fillna(0)
    pr = nx.pagerank(sg, weight="w")
    d["pagerank_station"] = [0.5 * (pr[s] + pr[s + 1]) for s in d.index]
    eb = nx.edge_betweenness_centrality(sg, weight="dist")
    d["edge_betweenness"] = [eb.get((s, s + 1), 0) + eb.get((s + 1, s), 0) for s in d.index]
    tpr = nx.pagerank(tg, weight="w")
    d["train_graph_score"] = [sum(tpr[t] for t in trains_through_section.get(s, ())) for s in d.index]
    for c in ("origin_freq", "size_triggered", "primary_minutes", "pagerank_station", "edge_betweenness", "train_graph_score"):
        d["rank_" + c] = d[c].rank(ascending=False, method="min")
    return d.reset_index()


def critical_trains(cascades, links, members, tg):
    """One row per train with cascade involvement and train-graph centralities."""
    m = members.set_index("run_idx")
    origin = cascades[cascades.origin_type == "primary"].groupby("origin_train").size().rename("origin_cascades")
    lk = links[links.cause == 4].copy()                                      # section links (a late leader delayed a follower)
    lk["leader_train"] = m.reindex(lk.parent).train.to_numpy()
    leader = lk.groupby("leader_train").size().rename("leader_links")
    caused = lk.groupby("leader_train").amount.sum().rename("caused_minutes")
    d = pd.concat([origin, leader, caused], axis=1).reindex(list(tg.nodes)).fillna(0)
    d["pagerank"] = pd.Series(nx.pagerank(tg, weight="w"))
    d["betweenness"] = pd.Series(nx.betweenness_centrality(tg, weight="dist"))
    d.index.name = "train"
    for c in d.columns:
        d["rank_" + c] = d[c].rank(ascending=False, method="min")
    return d.reset_index()


def rank_correlations(df, cols):
    """Spearman rank correlations between the listed metric columns."""
    return df[cols].corr(method="spearman")


def flag_artifacts(section_table, primary_events, section_pairs, artifact_trains, top=10, share=0.5):
    """Flag top-ranked sections that trace to the KNS-PJY quirk or are dominated by Shatabdi-artifact trains."""
    out = []
    rank_cols = [c for c in section_table.columns if c.startswith("rank_")]
    top_sections = {c: set(section_table.nsmallest(top, c).sec) for c in rank_cols}
    art = primary_events.assign(art=primary_events.train.astype(str).isin(artifact_trains))
    share_art = art.groupby("sec").apply(lambda g: g.added[g.art].sum() / max(g.added.sum(), 1e-9), include_groups=False)
    for name, sec in section_pairs.items():
        hits = [c for c, s in top_sections.items() if sec in s]
        out.append({"item": name, "section": sec, "in_top_of": hits, "flagged": bool(hits)})
    for sec, v in share_art.items():
        if v >= share:
            hits = [c for c, s in top_sections.items() if sec in s]
            out.append({"item": f"primary minutes {v:.0%} from Shatabdi-artifact trains", "section": int(sec), "in_top_of": hits, "flagged": bool(hits)})
    return pd.DataFrame(out)
