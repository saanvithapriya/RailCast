"""Item 4 (critical sections and trains) on a tiny network with known answers."""
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.analysis import criticality as cr


def small_graph():
    nodes = pd.DataFrame({"node": range(4)})
    edges = pd.DataFrame([dict(from_node=a, to_node=b, section=min(a, b), direction=1 if b > a else -1)
                          for a in range(3) for b in (a + 1,)] + [dict(from_node=b, to_node=a, section=a, direction=-1) for a in range(3) for b in (a + 1,)])
    cont = pd.DataFrame({"sec": [0, 0, 1, 1, 2, 2], "dir": [1, -1, 1, -1, 1, -1], "A_all": [0.1, 0.1, 0.5, 0.5, 0.1, 0.1]})
    return nodes, edges, cont


def test_station_graph_weights_and_distance():
    nodes, edges, cont = small_graph()
    g = cr.station_digraph(nodes, edges, cont)
    assert g.number_of_edges() == 6
    assert g[1][2]["w"] == 0.5 and g[1][2]["dist"] == pytest.approx(2.0)
    assert g[0][1]["dist"] == pytest.approx(10.0)


def test_section_ranks_follow_the_cascade_metrics():
    nodes, edges, cont = small_graph()
    sg = cr.station_digraph(nodes, edges, cont)
    casc = pd.DataFrame({"origin_type": ["primary", "primary", "primary", "external_entry"], "origin_sec": [1, 1, 2, 0], "size_runs": [4, 2, 1, 9]})
    prim = pd.DataFrame({"sec": [1, 1, 2], "added": [20.0, 30.0, 40.0], "train": ["a", "b", "c"]})
    tg = nx.DiGraph([("a", "b"), ("b", "c")]); nx.set_edge_attributes(tg, 1.0, "w"); nx.set_edge_attributes(tg, 1.0, "dist")
    t = cr.critical_sections(casc, prim, nodes, sg, tg, {0: ["a"], 1: ["b"], 2: ["c"]}, 3).set_index("sec")
    assert t.origin_freq.tolist() == [0, 2, 1] and t.size_triggered.tolist() == [0, 6, 1] and t.primary_minutes.tolist() == [0, 50, 40]
    assert t.external_origin_freq.tolist() == [1, 0, 0]
    assert t.rank_origin_freq.tolist() == [3, 1, 2] and t.rank_primary_minutes.tolist() == [3, 1, 2]
    assert t.pagerank_station.idxmax() == 1                         # the strongest contagion section sits in the middle


def test_train_table_counts_leaders_and_flags():
    members = pd.DataFrame({"run_idx": [0, 1, 2], "train": ["a", "b", "c"]})
    links = pd.DataFrame({"parent": [0, 0, 1], "cause": [4, 4, 4], "amount": [5.0, 7.0, 3.0]})
    casc = pd.DataFrame({"origin_type": ["primary"], "origin_train": ["a"]})
    tg = nx.DiGraph([("a", "b"), ("b", "c")]); nx.set_edge_attributes(tg, 1.0, "w"); nx.set_edge_attributes(tg, 1.0, "dist")
    t = cr.critical_trains(casc, links, members, tg).set_index("train")
    assert t.loc["a", "leader_links"] == 2 and t.loc["a", "caused_minutes"] == 12.0 and t.loc["a", "origin_cascades"] == 1
    assert t.loc["c", "leader_links"] == 0 and t.loc["a", "rank_leader_links"] == 1
    sec = pd.DataFrame({"sec": [5, 6], "rank_origin_freq": [1, 2], "rank_primary_minutes": [2, 1]})
    prim = pd.DataFrame({"sec": [5, 6], "added": [10.0, 10.0], "train": ["12003", "x"]})
    f = cr.flag_artifacts(sec, prim, {"KNS-PJY": 6}, {"12003", "12033"}, top=1)
    assert f.set_index("item").loc["KNS-PJY", "flagged"]               # section 6 is the top of rank_primary_minutes
    assert any("Shatabdi" in i for i in f.item)


def test_rank_correlation_is_one_for_identical_ranking():
    d = pd.DataFrame({"a": [3, 2, 1, 0], "b": [30, 20, 10, 0], "c": [0, 1, 2, 3]})
    c = cr.rank_correlations(d, ["a", "b", "c"])
    assert c.loc["a", "b"] == pytest.approx(1.0) and c.loc["a", "c"] == pytest.approx(-1.0)
