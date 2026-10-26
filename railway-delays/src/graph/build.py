"""Build all Phase 3 graph outputs for Corridor B.

    python -m src.graph.build [--scenario medium] [--no-groundtruth]

The time windows and the simulated data sources come from configs/splits.yaml (via src/splits.py), the only
place they are defined: train = all of 2024 + Jan-Aug 2025, validation = Sep-Oct 2025, test = Nov-Dec 2025.

Writes to data/processed/graph/:
  timetable-only:  station_nodes / station_edges / station_node_map (parquet), station_graph_timetable.pkl,
                   train_nodes / train_edges (parquet), train_graph_timetable.pkl
  SIMULATED delay attributes for the training window (use for development and training) and for all of 2025
  (ANALYSIS_ONLY, never for training); file names carry the window from the config.
and the ground-truth cascade tables (VALIDATION ONLY) inside the 2025 medium scenario folder.
"""
import argparse
import time
from pathlib import Path

from src import splits

from . import station_graph as sg
from . import train_graph as tg
from .delay_attrs import edge_delay_attributes, fold_train_delay_tables, node_delay_attributes
from .groundtruth import build_groundtruth_cascades
from .timetable import load_timetable

WINDOW_NAMES = {"train": "train", "all2025": "analysis_all_2025"}      # label -> name in configs/splits.yaml


def tag(label):
    return splits.window_tag(WINDOW_NAMES[label])


def main(scenario="medium", out_dir="data/processed/graph", groundtruth=True, groundtruth_dir="data/processed/sim/medium"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = splits.sources(scenario)
    t = time.time()
    tt = load_timetable()
    nodes, edges, node_map = sg.build_station_tables(tt)
    nodes.to_parquet(out / "station_nodes.parquet", index=False)
    edges.to_parquet(out / "station_edges.parquet", index=False)
    node_map.to_parquet(out / "station_node_map.parquet", index=False)
    sg.save_pickle(sg.build_station_graph(nodes, edges), out / "station_graph_timetable.pkl")
    tn, te = tg.build_train_tables(tt)
    tn.to_parquet(out / "train_nodes.parquet", index=False)
    te.to_parquet(out / "train_edges.parquet", index=False)
    sg.save_pickle(tg.build_train_graph(tn, te), out / "train_graph_timetable.pkl")
    print(f"timetable-only graphs built in {time.time() - t:.0f}s: {len(nodes)} nodes, {len(edges)} edges; "
          f"{len(tn)} trains, {len(te)} dependency edges", flush=True)
    for label, name in WINDOW_NAMES.items():
        a, b = splits.window(name)
        t = time.time()
        nd = node_delay_attributes(sources, a, b, tt.node_of)
        ed = edge_delay_attributes(sources, a, b, tt.node_of)
        nd.to_parquet(out / f"station_nodes_delay_{tag(label)}.parquet", index=False)
        ed.to_parquet(out / f"station_edges_delay_{tag(label)}.parquet", index=False)
        g = sg.build_station_graph(nodes, edges, nd, ed, label=tag(label))
        sg.save_pickle(g, out / f"station_graph_delays_{tag(label)}.pkl")
        print(f"delay attributes {label} ({a} .. {b}): {len(nd)} nodes, {len(ed)} edges in {time.time() - t:.0f}s", flush=True)
    # rolling-origin folds: tables built only from each fold's training range (fold 2's range equals the main
    # training window, which was built above, so it is skipped)
    for fname in splits.fold_names():
        if splits.fold(fname)["train"] == splits.window("train"):
            continue
        t = time.time()
        nd, ed = fold_train_delay_tables(fname, tt.node_of, scenario)
        ftag = splits.fold_tag(fname, "train")
        nd.to_parquet(out / f"station_nodes_delay_{ftag}.parquet", index=False)
        ed.to_parquet(out / f"station_edges_delay_{ftag}.parquet", index=False)
        print(f"delay attributes {fname} train ({splits.fold(fname)['train'][0]} .. {splits.fold(fname)['train'][1]}) in {time.time() - t:.0f}s", flush=True)
    if groundtruth:
        t = time.time()
        casc, runs = build_groundtruth_cascades(groundtruth_dir)
        print(f"ground-truth cascades (VALIDATION ONLY): {len(casc)} cascades, {len(runs)} run rows in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="medium")
    ap.add_argument("--out", default="data/processed/graph")
    ap.add_argument("--no-groundtruth", action="store_true")
    a = ap.parse_args()
    main(a.scenario, a.out, not a.no_groundtruth)
