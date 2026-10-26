"""Timetable-only view of Corridor B (no simulation outputs, no delays).

Reads only data/processed/{stations,trains,stops}.parquet (Phase 2) and the `return_train` field of
data/raw/trains.json. Everything built on this module is safe to use without leakage.

Representation of Prayagraj: ALD, NYN and COI are merged into ONE node named PRYG (as in the simulator);
`Timetable.node_members` keeps the mapping back to the original station codes.

Every train is treated as running daily (the timetable has no day-of-week information), so "per day"
counts are counts of trains.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.config import load_config
from src.simulation.network import build_links, build_network, build_spec, load_return_map

FAST_MAX_PRIORITY = 3          # Rajdhani, Duronto, Shatabdi, Superfast
TYPES = ["Rajdhani", "Duronto", "Shatabdi", "Superfast", "Mail/Express", "Passenger", "other"]


@dataclass
class Timetable:
    net: object                  # simulation.engine.Net (node names, junction flags, section km)
    specs: dict                  # train -> TrainSpec (visits, hops, scheduled times in a local frame)
    node_of: dict                # original station code -> node index
    node_members: list           # node index -> list of original station codes
    stations: pd.DataFrame       # Phase 2 stations table (110 rows)
    trains: pd.DataFrame         # Phase 2 trains table without slip entries (225 rows)
    link_stats: dict
    link_rule: dict              # train -> rule that created its rake link (return_train / odd_even)
    cfg: dict


def load_timetable(processed_dir="data/processed", raw_dir="data/raw", cfg=None):
    cfg = cfg or load_config()
    p = Path(processed_dir)
    stations = pd.read_parquet(p / "stations.parquet")
    trains = pd.read_parquet(p / "trains.parquet")
    stops = pd.read_parquet(p / "stops.parquet")
    trains = trains[~trains.is_slip].reset_index(drop=True)
    stops = stops[stops.train.isin(trains.number)]
    net, node_of, coords = build_network(stations, stops, cfg)
    members = [[] for _ in net.node_names]
    for code, i in node_of.items():
        members[i].append(code)
    # keep the chain order of members: stations table order is by train count, so sort by chain position later
    specs = {}
    for _, tr in trains.iterrows():
        g = stops[stops.train == tr["number"]].sort_values("sequence")
        specs[tr["number"]] = build_spec(tr, g, net, node_of, coords, cfg)
    rmap = load_return_map(raw_dir)
    stats = build_links(specs, rmap, cfg)
    rule = {tn: ("return_train" if sp.feeder[0] == rmap.get(tn) else "odd_even") for tn, sp in specs.items() if sp.feeder}
    return Timetable(net, specs, node_of, members, stations, trains, stats, rule, cfg)


def section_passages(tt):
    """One row per (train, section): the scheduled entry time and scheduled run time (minutes).

    entry_min is in the train's local frame (minutes from 00:00 of its entry date); entry_mod is the
    time of day. Trains run daily, so each row is a daily event. Off-corridor legs are not included.
    """
    rows = []
    for tn, sp in tt.specs.items():
        for k, h in enumerate(sp.hops):
            if h.kind != "section":
                continue
            acc = sp.visits[k].sched_dep
            for s, sh in zip(h.sections, h.shares):
                rows.append((tn, sp.type, sp.priority, s, h.direction, acc, acc % 1440.0, sh))
                acc += sh
    return pd.DataFrame(rows, columns=["train", "type", "priority", "section", "direction", "entry_min", "entry_mod", "run_min"])


def node_visits(tt):
    """One row per (train, node visit) with scheduled times and whether the visit is a real halt."""
    rows = []
    for tn, sp in tt.specs.items():
        for k, v in enumerate(sp.visits):
            nxt = sp.hops[k] if k < len(sp.hops) else None
            d = nxt.direction if (nxt is not None and nxt.kind == "section") else 0
            rows.append((tn, sp.type, sp.priority, k, v.node, v.sched_arr, v.sched_dep, v.sched_arr % 1440.0,
                         v.sched_dep % 1440.0, bool(v.halt), v.sched_dep - v.sched_arr, d))
    return pd.DataFrame(rows, columns=["train", "type", "priority", "visit", "node", "sched_arr", "sched_dep", "arr_mod",
                                       "dep_mod", "halt", "dwell", "direction"])
