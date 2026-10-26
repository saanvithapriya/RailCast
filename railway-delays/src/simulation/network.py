"""Build the section network and per-train specs from the Phase 2 tables (SIMULATED data).

Choices (see design document):
* Consecutive core stations on the chain are sections; ALD, NYN and COI are merged into one node.
* Scheduled time is used only between real halts (and entry/exit) and split over sections by distance;
  pass-through scheduled times are interpolated, the raw times are kept in `rows`.
* Consecutive core stops more than `off_corridor_hop_km` apart form a fixed-time off-corridor leg.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import Hop, Net, TrainSpec, Visit

EPS = 1e-9


def haversine_km(lat0, lon0, lat1, lon1):
    p0, p1 = math.radians(lat0), math.radians(lat1)
    a = math.sin((p1 - p0) / 2) ** 2 + math.cos(p0) * math.cos(p1) * math.sin(math.radians(lon1 - lon0) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def build_chain(stops):
    """Station order NDLS -> MGS from the train that covers the most distinct core stations."""
    by = {t: g.sort_values("sequence").station.tolist() for t, g in stops.groupby("train")}
    full = {t: c for t, c in by.items() if "NDLS" in c and "MGS" in c}
    best = max(sorted(full), key=lambda t: len(set(full[t])))
    chain = list(dict.fromkeys(full[best]))
    return chain[::-1] if chain.index("NDLS") > chain.index("MGS") else chain


def build_network(stations, stops, cfg):
    """Return (Net, node_of_station dict, node_coords list)."""
    chain = build_chain(stops)
    cluster = set(cfg["network"]["cluster_stations"])
    cname = cfg["network"]["cluster_name"]
    names, members = [], []
    for c in chain:
        n = cname if c in cluster else c
        if names and names[-1] == n:
            members[-1].append(c)
        elif n == cname and cname in names:
            raise ValueError("cluster stations are not contiguous on the chain")
        else:
            names.append(n)
            members.append([c])
    st = stations.set_index("code")
    coords = [(st.lat[m[0]], st.lon[m[0]]) for m in members]
    junction = [bool(any(st.is_junction[c] for c in m)) for m in members]
    km = [max(0.2, haversine_km(*coords[i], *coords[i + 1])) for i in range(len(names) - 1)]
    mv = cfg["movement"]
    over = mv.get("headway_overrides") or {}
    headway = [float(over.get(f"{names[i]}-{names[i + 1]}", mv["headway_min"])) for i in range(len(km))]
    inc = cfg["incidents"]

    def nfac(i):
        return inc["node_factor"].get(names[i], inc["junction_factor"] if junction[i] else 1.0)

    sec_factor = [max(nfac(i), nfac(i + 1)) for i in range(len(km))]
    congested = set(cfg["freight"]["congested_nodes"])
    sec_cong = [(names[i] in congested) or (names[i + 1] in congested) for i in range(len(km))]
    fi = cfg["weather"]["fog_intensity"]
    i_cnb, i_pr = names.index("CNB"), names.index(cname)
    fog = [fi["to_CNB"] if i < i_cnb else fi["to_PRYG"] if i < i_pr else fi["beyond"] for i in range(len(km))]
    net = Net(names, junction, km, headway, sec_factor, sec_cong, fog)
    node_of = {c: i for i, m in enumerate(members) for c in m}
    return net, node_of, coords


def _group_visits(nodes):
    """Index pairs (first, last) of consecutive rows sharing a node."""
    groups, start = [], 0
    for i in range(1, len(nodes) + 1):
        if i == len(nodes) or nodes[i] != nodes[start]:
            groups.append((start, i - 1))
            start = i
    return groups


def build_spec(tr, rows, net, node_of, coords, cfg):
    """Build one TrainSpec. `tr` is a trains-table row, `rows` its stops sorted by sequence."""
    mv = cfg["movement"]
    nodes = [node_of[s] for s in rows.station]
    o = float(tr["origin_departure_min"])
    arr_abs = o + rows.minutes_from_start.to_numpy()
    dep_abs = o + rows.dep_min_from_start.to_numpy()
    shift = math.floor(arr_abs[0] / 1440.0) * 1440.0
    la, ld = arr_abs - shift, dep_abs - shift
    raw_arr = np.where(rows.scheduled_arrival.isna().to_numpy(), np.nan, la)
    raw_dep = np.where(rows.scheduled_departure.isna().to_numpy(), np.nan, ld)
    is_halt = rows.is_halt.to_numpy()

    groups = _group_visits(nodes)
    nv = len(groups)
    v_node = [nodes[a] for a, _ in groups]
    v_arr = [float(la[a]) for a, _ in groups]
    v_dep = [float(ld[b]) for _, b in groups]
    v_halt = [bool(is_halt[a:b + 1].any()) for a, b in groups]

    hops, is_leg = [], []
    for k in range(nv - 1):
        pa, pb = v_node[k], v_node[k + 1]
        km_line = haversine_km(*coords[pa], *coords[pb])
        if km_line > cfg["network"]["off_corridor_hop_km"] or pa == pb:
            hops.append(Hop("leg", leg_minutes=max(0.0, v_arr[k + 1] - v_dep[k])))
            is_leg.append(True)
        else:
            d = 1 if pb > pa else -1
            secs = tuple(range(pa, pb)) if d == 1 else tuple(range(pa - 1, pb - 1, -1))
            hops.append(Hop("section", sections=secs, direction=d))
            is_leg.append(False)

    # anchors: first/last visits, real halts, fixed dwell, and the ends of legs
    dwell = [v_dep[k] - v_arr[k] for k in range(nv)]
    anchor = [k == 0 or k == nv - 1 or v_halt[k] or dwell[k] > EPS for k in range(nv)]
    for k, leg in enumerate(is_leg):
        if leg:
            anchor[k] = anchor[k + 1] = True
    sched_arr, sched_dep = list(v_arr), list(v_dep)
    anchors = [k for k in range(nv) if anchor[k]]
    for a, b in zip(anchors, anchors[1:]):
        if is_leg[a]:
            continue
        secs = [s for k in range(a, b) for s in hops[k].sections]
        w = np.array([net.section_km[s] for s in secs])
        total = max(v_arr[b] - v_dep[a], 0.0)
        share = total * w / w.sum()
        pos = 0
        for k in range(a, b):
            n = len(hops[k].sections)
            hops[k].shares = tuple(float(x) for x in share[pos:pos + n])
            pos += n
            if k + 1 < b:                                   # pass-through visit: interpolated time
                t = v_dep[a] + float(share[:pos].sum())
                sched_arr[k + 1] = sched_dep[k + 1] = t

    f = mv["min_halt_factor"]
    visits = []
    for k in range(nv):
        mh = 0.0
        if v_halt[k] and dwell[k] > EPS:
            mh = min(dwell[k], max(1.0, float(round(f * dwell[k]))))
        visits.append(Visit(v_node[k], sched_arr[k], sched_dep[k], v_halt[k], mh))

    rowinfo = []
    for k, (a, b) in enumerate(groups):
        n = b - a + 1
        for j, r in enumerate(range(a, b + 1)):
            interp = (not anchor[k]) and n == 1
            rowinfo.append((rows.station.iloc[r], int(rows.sequence.iloc[r]), k, j, n,
                            sched_arr[k] if interp else float(la[r]), sched_dep[k] if interp else float(ld[r]),
                            float(raw_arr[r]), float(raw_dep[r])))
    span = max(float(tr["terminus_arrival_min"]) - o, 1.0)
    return TrainSpec(
        train=str(tr["number"]), type=tr["type"], priority=mv["priority"].get(tr["type"], 4),
        slack=mv["slack"].get(tr["type"], 0.06), visits=visits, hops=hops,
        entered_mid_route=bool(tr["entered_mid_route"]), exits_mid_route=bool(tr["exits_mid_route"]),
        extra={"rows": rowinfo, "shift_days": shift / 1440.0, "origin_local": o - shift,
               "terminus_local": float(tr["terminus_arrival_min"]) - shift,
               "journey_fraction": float(np.clip(rows.minutes_from_start.iloc[0] / span, 0.0, 1.0))})


def scheduled_pair_gaps(specs, net, exit_gap, window_min=30.0):
    """Spacing the timetable itself uses between pairs of trains on a section (same direction).

    The timetable has no day-of-week information, so every train runs daily and many trains share a
    slot. Pairs scheduled closer than the headway are assumed feasible (parallel lines, or trains that
    do not really run on the same day). For such pairs the required spacing is the scheduled spacing:
    returns {(sec, dir): {(follower, leader): (min entry gap, min arrival gap)}}.
    """
    ent = {}
    for tn, sp in specs.items():
        for k, h in enumerate(sp.hops):
            if h.kind != "section":
                continue
            acc = sp.visits[k].sched_dep
            for s, sh in zip(h.sections, h.shares):
                ent.setdefault((s, h.direction), []).append((acc % 1440.0, sh, tn))
                acc += sh
    pairs = {}
    for key, lst in ent.items():
        lst.sort()
        hw = net.headway[key[0]]
        n = len(lst)
        d = {}
        for i in range(n):
            ti, di, a = lst[i]
            for step in range(1, n):
                tj, dj, b = lst[(i + step) % n]
                gap = (tj - ti) % 1440.0
                if gap >= window_min:
                    break
                if a == b:
                    continue
                arr_gap = gap + dj - di
                if gap < hw or arr_gap < exit_gap:
                    d[(b, a)] = (min(hw, gap), min(exit_gap, arr_gap))        # b follows a
                    d.setdefault((a, b), (min(hw, gap), min(exit_gap, -arr_gap)))  # a follows b if the order flips
        if d:
            pairs[key] = d
    return pairs


def load_return_map(raw_dir="data/raw"):
    feats = json.load(open(Path(raw_dir) / "trains.json"))["features"]
    return {f["properties"]["number"]: f["properties"].get("return_train") for f in feats}


def build_links(specs, return_map, cfg):
    """Attach `feeder = (partner, k, gap)` to each spec. Returns link statistics.

    The partner is `return_train` first, consecutive odd/even numbers as fallback. The link needs
    the scheduled turnaround to be between min_turnaround and max_turnaround_hours.
    """
    rk = cfg["rake"]
    lo, hi = rk["min_turnaround_min"], rk["max_turnaround_hours"] * 60.0
    stats = {"return_train": 0, "odd_even": 0, "no_partner": 0, "turnaround_out_of_window": 0}
    for tn, sp in specs.items():
        partner, rule = None, None
        rt = return_map.get(tn)
        if rt and rt in specs and rt != tn:
            partner, rule = rt, "return_train"
        elif tn.isdigit():
            n = int(tn)
            cand = str(n + 1 if n % 2 else n - 1).zfill(len(tn))
            if cand in specs:
                partner, rule = cand, "odd_even"
        if partner is None:
            stats["no_partner"] += 1
            continue
        gap0 = sp.extra["origin_local"] - specs[partner].extra["terminus_local"]
        k = max(0, math.ceil((lo - gap0) / 1440.0))
        gap = gap0 + k * 1440.0
        if gap > hi:
            stats["turnaround_out_of_window"] += 1
            continue
        sp.feeder = (partner, k, gap)
        stats[rule] += 1
    stats["linked"] = stats["return_train"] + stats["odd_even"]
    stats["trains"] = len(specs)
    return stats


def load_specs(processed_dir="data/processed", raw_dir="data/raw", cfg=None):
    """Load Phase 2 tables and return (net, specs, link_stats, cfg). Slip entries are dropped."""
    from .config import load_config

    cfg = cfg or load_config()
    p = Path(processed_dir)
    stations = pd.read_parquet(p / "stations.parquet")
    trains = pd.read_parquet(p / "trains.parquet")
    stops = pd.read_parquet(p / "stops.parquet")
    trains = trains[~trains.is_slip]
    stops = stops[stops.train.isin(trains.number)]
    net, node_of, coords = build_network(stations, stops, cfg)
    specs = {}
    for _, tr in trains.iterrows():
        g = stops[stops.train == tr["number"]].sort_values("sequence")
        specs[tr["number"]] = build_spec(tr, g, net, node_of, coords, cfg)
    stats = build_links(specs, load_return_map(raw_dir), cfg)
    return net, specs, stats, cfg
