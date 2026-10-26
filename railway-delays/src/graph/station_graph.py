"""Station graph: nodes = 108 nodes (110 stations, ALD/NYN/COI merged into PRYG), directed edges between
consecutive nodes on the main-line chain (both directions). Timetable-only attributes here; delay-based
attributes (SIMULATED) are in delay_attrs.py and are attached only when explicitly requested.
"""
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.simulation.network import haversine_km

from .timetable import FAST_MAX_PRIORITY, TYPES, node_visits, section_passages

CLUSTER_DISPLAY = "Prayagraj area (Prayagraj Jn + Naini Jn + Prayagraj Chheoki)"


def build_station_tables(tt):
    """Return (nodes, edges, node_map): timetable-only tables.

    nodes: one row per node. node_map: one row per original station (110) pointing to its node.
    edges: directed, 2 x 107 rows. All counts are per day (every train runs daily).
    """
    net = tt.net
    st = tt.stations.set_index("code")
    visits = node_visits(tt)
    passages = section_passages(tt)

    members = []
    for i, m in enumerate(tt.node_members):
        members.append(m)
    km_cum = np.r_[0.0, np.cumsum(net.section_km)]
    rows = []
    for i, name in enumerate(net.node_names):
        m = members[i]
        first = st.loc[m[0]]
        v = visits[visits.node == i]
        row = {
            "node": i, "node_name": name, "original_codes": "|".join(m), "n_original_stations": len(m),
            "display_name": CLUSTER_DISPLAY if len(m) > 1 else first.display_name,
            "lat": float(first.lat), "lon": float(first.lon), "is_junction": bool(net.node_junction[i]),
            "km_from_new_delhi": float(km_cum[i]),
            "trains_per_day": int(v.train.nunique()),
            "halts_per_day": int(v.halt.sum()),
            "trains_down_per_day": int(v[v.direction == 1].train.nunique()),
            "trains_up_per_day": int(v[v.direction == -1].train.nunique()),
        }
        tv = v.drop_duplicates("train")
        for t in TYPES:
            row[f"share_{t}"] = float((tv.type == t).mean()) if len(tv) else 0.0
        rows.append(row)
    nodes = pd.DataFrame(rows)

    node_map = pd.DataFrame({
        "station_code": list(tt.node_of), "node": list(tt.node_of.values()),
    })
    node_map["node_name"] = node_map.node.map(dict(enumerate(net.node_names)))
    node_map["display_name"] = node_map.station_code.map(st.display_name)
    node_map = node_map.sort_values(["node", "station_code"]).reset_index(drop=True)

    erows = []
    for direction in (1, -1):
        for sec in range(net.n_sections):
            a, b = (sec, sec + 1) if direction == 1 else (sec + 1, sec)
            p = passages[(passages.section == sec) & (passages.direction == direction)]
            hours = np.bincount((p.entry_mod // 60).astype(int).clip(0, 23), minlength=24)
            fast = p.priority <= FAST_MAX_PRIORITY
            row = {
                "from_node": a, "to_node": b, "from_name": net.node_names[a], "to_name": net.node_names[b],
                "section": sec, "direction": direction, "km": float(net.section_km[sec]),
                "trains_per_day": int(len(p)),
                "mean_sched_run_min": float(p.run_min.mean()) if len(p) else np.nan,
                "median_sched_run_min": float(p.run_min.median()) if len(p) else np.nan,
                "n_fast": int(fast.sum()), "n_slow": int((~fast).sum()),
                "share_fast": float(fast.mean()) if len(p) else np.nan,
                "share_slow": float((~fast).mean()) if len(p) else np.nan,
            }
            for h in range(24):
                row[f"trains_hour_{h:02d}"] = int(hours[h])
            erows.append(row)
    edges = pd.DataFrame(erows)
    edges.insert(0, "edge_id", range(len(edges)))
    return nodes, edges, node_map


def build_station_graph(nodes, edges, node_delay=None, edge_delay=None, label=None):
    """NetworkX DiGraph. Optional SIMULATED delay tables are merged in as extra attributes."""
    g = nx.DiGraph()
    nd = node_delay.set_index("node") if node_delay is not None else None
    for r in nodes.itertuples(index=False):
        attrs = r._asdict()
        if nd is not None and r.node in nd.index:
            attrs.update({f"sim_{k}": v for k, v in nd.loc[r.node].to_dict().items()})
        g.add_node(int(r.node), **attrs)
    ed = edge_delay.set_index(["from_node", "to_node"]) if edge_delay is not None else None
    for r in edges.itertuples(index=False):
        attrs = r._asdict()
        if ed is not None and (r.from_node, r.to_node) in ed.index:
            attrs.update({f"sim_{k}": v for k, v in ed.loc[(r.from_node, r.to_node)].to_dict().items()})
        g.add_edge(int(r.from_node), int(r.to_node), **attrs)
    g.graph["description"] = "Corridor B station graph (108 nodes, PRYG = ALD+NYN+COI)"
    g.graph["simulated_delay_attributes"] = bool(node_delay is not None or edge_delay is not None)
    if label:
        g.graph["delay_label"] = label
    return g


def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
