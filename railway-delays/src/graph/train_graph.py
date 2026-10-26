"""Train-dependency graph (timetable only): nodes are trains, edges are scheduled dependencies.

Edge types:
* rake            directed feeder -> next service of the rake; attrs: rule (return_train / odd_even), gap_min
* section_sharing undirected; the two trains are scheduled to enter the same section in the same direction
                  within `window_min` minutes (circular over the day, since every train runs daily);
                  count = number of sections where this happens, mean_gap_min = mean scheduled entry gap
* overtake_station directed X -> Y; X passes a node while Y has a scheduled real halt there (a scheduled
                  overtaking opportunity); count = number of nodes, mean_gap_min = mean minutes between X
                  passing and Y leaving
* overtake_section directed X -> Y; X enters a section after Y but is scheduled to arrive before Y
                  (a scheduled crossing inside a section); count = number of sections, mean_gap_min = mean
                  minutes by which X arrives before Y

Timetable only: nothing here reads simulation outputs.
"""
import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from .timetable import node_visits, section_passages

DAY = 1440.0


def _circ(a, b):
    """Forward distance from time a to time b around the clock, in [0, 1440)."""
    return (b - a) % DAY


def rake_edges(tt):
    rows = []
    for tn, sp in tt.specs.items():
        if sp.feeder is None:
            continue
        ftrain, k, gap = sp.feeder
        rule = "return_train" if tt.link_rule.get(tn) == "return_train" else "odd_even"
        rows.append((ftrain, tn, "rake", 1, float(gap), rule))
    return pd.DataFrame(rows, columns=["source", "target", "type", "count", "mean_gap_min", "rule"])


def section_sharing_edges(tt, window_min=10.0):
    p = section_passages(tt)
    acc = defaultdict(lambda: [0, 0.0])
    for (sec, d), g in p.groupby(["section", "direction"]):
        t = g.entry_mod.to_numpy()
        names = g.train.to_numpy()
        order = np.argsort(t)
        t, names = t[order], names[order]
        n = len(t)
        for i in range(n):
            for step in range(1, n):
                j = (i + step) % n
                gap = _circ(t[i], t[j])
                if gap >= window_min:
                    break
                if names[i] == names[j]:
                    continue
                key = tuple(sorted((names[i], names[j])))
                acc[key][0] += 1
                acc[key][1] += gap
    rows = [(a, b, "section_sharing", c, s / c, "") for (a, b), (c, s) in acc.items()]
    return pd.DataFrame(rows, columns=["source", "target", "type", "count", "mean_gap_min", "rule"])


def overtake_station_edges(tt):
    v = node_visits(tt)
    acc = defaultdict(lambda: [0, 0.0])
    for (node, d), g in v[v.direction != 0].groupby(["node", "direction"]):
        halts = g[g.dwell > 0]
        for h in halts.itertuples():
            for x in g.itertuples():
                if x.train == h.train:
                    continue
                # X passes (or stops at) the node while Y sits there: X's time at the node inside Y's halt window
                inside = _circ(h.arr_mod, x.arr_mod) <= h.dwell
                if inside:
                    acc[(x.train, h.train)][0] += 1
                    acc[(x.train, h.train)][1] += h.dwell - _circ(h.arr_mod, x.arr_mod)
    rows = [(a, b, "overtake_station", c, s / c, "") for (a, b), (c, s) in acc.items()]
    return pd.DataFrame(rows, columns=["source", "target", "type", "count", "mean_gap_min", "rule"])


def overtake_section_edges(tt, window_min=30.0):
    p = section_passages(tt)
    acc = defaultdict(lambda: [0, 0.0])
    for (sec, d), g in p.groupby(["section", "direction"]):
        t = g.entry_mod.to_numpy()
        run = g.run_min.to_numpy()
        names = g.train.to_numpy()
        n = len(t)
        for i in range(n):
            for j in range(n):
                if i == j or names[i] == names[j]:
                    continue
                gap = _circ(t[i], t[j])            # j enters gap minutes after i
                if gap <= 0 or gap >= window_min:
                    continue
                arr_gap = gap + run[j] - run[i]    # j arrives arr_gap after i
                if arr_gap < 0:                    # j overtakes i inside the section
                    acc[(names[j], names[i])][0] += 1
                    acc[(names[j], names[i])][1] += -arr_gap
    rows = [(a, b, "overtake_section", c, s / c, "") for (a, b), (c, s) in acc.items()]
    return pd.DataFrame(rows, columns=["source", "target", "type", "count", "mean_gap_min", "rule"])


def build_train_tables(tt, window_min=10.0):
    """(nodes, edges): trains and all dependency edges."""
    nodes = pd.DataFrame([{
        "train": tn, "name": tt.trains.set_index("number").name[tn], "type": sp.type, "priority": sp.priority,
        "entered_mid_route": sp.entered_mid_route, "exits_mid_route": sp.exits_mid_route,
        "n_visits": len(sp.visits), "n_sections": int(sum(len(h.sections) for h in sp.hops if h.kind == "section")),
        "has_off_corridor_leg": any(h.kind == "leg" for h in sp.hops),
        "direction": int(np.sign(sum(h.direction for h in sp.hops if h.kind == "section"))),
    } for tn, sp in tt.specs.items()])
    edges = pd.concat([rake_edges(tt), section_sharing_edges(tt, window_min), overtake_station_edges(tt),
                       overtake_section_edges(tt)], ignore_index=True)
    return nodes, edges


def build_train_graph(nodes, edges):
    """MultiDiGraph; section_sharing edges are stored once with undirected=True."""
    g = nx.MultiDiGraph()
    for r in nodes.itertuples(index=False):
        g.add_node(r.train, **r._asdict())
    for r in edges.itertuples(index=False):
        g.add_edge(r.source, r.target, key=r.type, type=r.type, count=int(r.count), mean_gap_min=float(r.mean_gap_min),
                   rule=r.rule, undirected=(r.type == "section_sharing"))
    g.graph["description"] = "Corridor B train-dependency graph (timetable only)"
    return g


def degree_summary(edges, nodes):
    """Degree of every train per edge type and in total (distinct neighbours)."""
    deg = {}
    for typ, g in edges.groupby("type"):
        nb = defaultdict(set)
        for a, b in zip(g.source, g.target):
            nb[a].add(b)
            nb[b].add(a)
        deg[typ] = pd.Series({t: len(nb.get(t, ())) for t in nodes.train})
    allnb = defaultdict(set)
    for a, b in zip(edges.source, edges.target):
        allnb[a].add(b)
        allnb[b].add(a)
    deg["all"] = pd.Series({t: len(allnb.get(t, ())) for t in nodes.train})
    return pd.DataFrame(deg)
