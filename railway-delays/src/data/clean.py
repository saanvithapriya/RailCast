"""Reusable cleaning functions for the datameet timetable (Corridor B).

Raw files in data/raw/ are only ever read. Nothing here writes to them.

Assumptions (also recorded in reports/phase2_cleaning_log.md):
* The raw file has no day-of-week information: every train is treated as running daily.
* `day` in schedules.json is the 1-based day of the journey on which the row's times fall.
  A halt that crosses midnight (arrival 23:50, departure 00:20) has one `day` value, so the
  departure is rolled over to the next day.
* A row missing only one of arrival/departure (not origin/terminus) is filled from the other,
  i.e. a zero-minute halt is assumed. Rows missing both are dropped. No other time is invented.
"""
import json
from pathlib import Path

import pandas as pd

ANCHORS = ["NDLS", "TDL", "ETW", "CNB", "ALD", "MGS"]
MIN_CORRIDOR_TRAINS = 20
MIN_CORE_STOPS = 2
OFF_LINE_HOP_KM = 60

# Core-set stations (>=20 corridor trains) that are not on the Delhi-Mughalsarai main line.
OFFLINE_STATIONS = {
    "DLI": "Old Delhi: Delhi-Shahdara side loop",
    "DSA": "Delhi Shahdara: Delhi-Shahdara side loop",
    "VVB": "Viveka Vihar: Delhi-Shahdara side loop",
    "CNJO": "Kanpur yard / cabin",
    "GMC": "Kanpur yard / cabin",
    "CPNL": "Kanpur yard / cabin",
    "CPB": "Kanpur yard / cabin",
    "PMPR": "Kanpur-Lucknow line",
    "MGW": "Kanpur-Lucknow line",
    "ON": "Kanpur-Lucknow line",
    "LKO": "Kanpur-Lucknow line",
    "NEWC": "Varanasi side, beyond Mughalsarai",
    "VYN": "Varanasi side, beyond Mughalsarai",
    "KEI": "Varanasi side, beyond Mughalsarai",
    "BSB": "Varanasi side, beyond Mughalsarai",
}

# Current names for stations renamed after the timetable was compiled (original code stays the key).
DISPLAY_NAMES = {
    "ALD": "Prayagraj Jn",
    "MGS": "Pt. Deen Dayal Upadhyaya Jn",
    "COI": "Prayagraj Chheoki",
    "NYN": "Naini Jn",
}
KNOWN_JUNCTIONS = {"GZB"}  # Ghaziabad is a junction but its raw name lacks "JN"


# --------------------------------------------------------------------------- loading
def load_raw(raw_dir="data/raw"):
    """Return (schedules, stations, trains_meta) as DataFrames, as in the raw files."""
    raw_dir = Path(raw_dir)
    sched = pd.DataFrame(json.load(open(raw_dir / "schedules.json")))
    rows = []
    for f in json.load(open(raw_dir / "stations.json"))["features"]:
        p = dict(f["properties"])
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") if isinstance(geom, dict) else None
        p["lon"], p["lat"] = (coords[0], coords[1]) if coords else (None, None)
        rows.append(p)
    stations = pd.DataFrame(rows)
    trains_meta = pd.DataFrame([f["properties"] for f in json.load(open(raw_dir / "trains.json"))["features"]])
    return sched, stations, trains_meta


# --------------------------------------------------------------------------- corridor definition
def drop_duplicate_stops(sched):
    """Drop exact duplicate stop rows (ignoring the row id). Returns (clean, n_dropped)."""
    cols = [c for c in sched.columns if c != "id"]
    keep = ~sched.duplicated(subset=cols, keep="first")
    return sched[keep].copy(), int((~keep).sum())


def add_sequence(sched):
    """Sort by train and id (id is consecutive within a train); add 0-based `seq` and `last_seq`."""
    s = sched.sort_values(["train_number", "id"]).copy()
    s["seq"] = s.groupby("train_number").cumcount()
    s["last_seq"] = s.groupby("train_number").seq.transform("max")
    return s


def corridor_trains(sched_seq, anchors=ANCHORS):
    """Trains stopping at 2+ distinct anchors, with first/last anchor position."""
    a = sched_seq[sched_seq.station_code.isin(anchors)]
    g = a.groupby("train_number").agg(
        n_anchors=("station_code", "nunique"), lo=("seq", "min"), hi=("seq", "max"))
    return g[g.n_anchors >= 2]


def corridor_slice(sched_seq, ct):
    """Stops between each corridor train's first and last anchor, inclusive."""
    cs = sched_seq.merge(ct[["lo", "hi"]], left_on="train_number", right_index=True)
    return cs[(cs.seq >= cs.lo) & (cs.seq <= cs.hi)]


def station_train_counts(corridor_stops):
    return corridor_stops.groupby("station_code").train_number.nunique()


def is_junction_guess(name, code, anchors=ANCHORS):
    """Guess: name contains JN/JUNCTION, an anchor, or in KNOWN_JUNCTIONS.

    Neighbour counts are not used: express trains skip stops, so consecutive-stop
    adjacency over-counts neighbours.
    """
    n = str(name).upper()
    return bool(" JN" in n or "JUNCTION" in n or code in anchors or code in KNOWN_JUNCTIONS)


def main_line_chain(core_stops):
    """Order core stations along the main line and list branch/loop stops.

    The chain is the core-stop sequence of the train that stops at NDLS and MGS and has the
    most distinct core stops (oriented NDLS -> MGS). Stations not on it are branch/loop stops.
    Returns (chain, branch_stops, inconsistent_trains): the last value lists trains whose
    on-chain stops are not monotone along the chain (they reverse or use a loop).
    """
    by_train = {t: g.sort_values("seq").station_code.tolist() for t, g in core_stops.groupby("train_number")}
    full = [c for c in by_train.values() if "NDLS" in c and "MGS" in c]
    best = max(full, key=lambda c: len(set(c)))
    chain = list(dict.fromkeys(best))
    if chain.index("NDLS") > chain.index("MGS"):
        chain = chain[::-1]
    pos = {c: i for i, c in enumerate(chain)}
    branch = sorted(set(core_stops.station_code) - set(chain))
    bad = []
    for tr, codes in by_train.items():
        p = [pos[c] for c in codes if c in pos]
        inc = all(a <= b for a, b in zip(p, p[1:]))
        dec = all(a >= b for a, b in zip(p, p[1:]))
        if not (inc or dec):
            bad.append(tr)
    return chain, branch, bad


# --------------------------------------------------------------------------- train types
TYPE_RULES = (
    "1. name contains RAJDHANI -> Rajdhani\n"
    "2. name contains SHATABDI -> Shatabdi\n"
    "3. name contains DURONTO -> Duronto\n"
    "4. number starts with 5 (passenger series), or name contains PASSENGER/MEMU/DEMU -> Passenger\n"
    "5. number starts with 12 or 22 (superfast series), or name contains ' SF'/SUPERFAST -> Superfast "
    "(this includes Sampark Kranti and Garib Rath, which carry superfast numbers)\n"
    "6. number starts with 1 or 2, or name contains EXP/MAIL -> Mail/Express\n"
    "7. anything else (e.g. 0xxxx specials) -> other"
)


def classify_train(number, name):
    n = str(name).upper()
    num = str(number)
    if "RAJDHANI" in n:
        return "Rajdhani"
    if "SHATABDI" in n:
        return "Shatabdi"
    if "DURONTO" in n:
        return "Duronto"
    if num.startswith("5") or any(k in n for k in ("PASSENGER", "MEMU", "DEMU")):
        return "Passenger"
    if num.startswith(("12", "22")) or " SF" in n or "SUPERFAST" in n:
        return "Superfast"
    if num.startswith(("1", "2")) or "EXP" in n or "MAIL" in n:
        return "Mail/Express"
    return "other"


# --------------------------------------------------------------------------- time cleaning
def _tod_minutes(col):
    """'HH:MM:SS' -> minutes since midnight; the string 'None' -> NaN."""
    parts = col.where(col.ne("None")).str.split(":", expand=True)
    return pd.to_numeric(parts[0]) * 60 + pd.to_numeric(parts[1]) + pd.to_numeric(parts[2]) / 60


class CleaningLog:
    """Collects (step, message, counts) entries."""

    def __init__(self):
        self.entries = []

    def add(self, step, message, **counts):
        self.entries.append({"step": step, "message": message, "counts": counts})

    def get(self, step):
        return next(e["counts"] for e in self.entries if e["step"] == step)


def apply_time_rules(full, core_codes, log):
    """Apply the 'None' rules and build absolute minutes. `full` = full routes of corridor trains.

    Returns the kept rows with columns arr_abs/dep_abs (minutes from the start of day 1 of the
    journey) and day_offset. Dropped rows are logged.
    """
    f = full.copy()
    on_core = f.station_code.isin(core_codes)
    arr_none, dep_none = f.arrival.eq("None"), f.departure.eq("None")
    is_origin, is_term = f.seq.eq(0), f.seq.eq(f.last_seq)

    log.add("none_legit", "origin has no arrival / terminus has no departure (legitimate, kept as null)",
            origin_no_arrival=int((arr_none & ~dep_none & is_origin).sum()),
            terminus_no_departure=int((dep_none & ~arr_none & is_term).sum()))

    both = arr_none & dep_none
    log.add("none_both", "stop with neither arrival nor departure: dropped (no time to use)",
            dropped=int(both.sum()), of_which_core=int((both & on_core).sum()),
            of_which_origin_or_terminus=int((both & (is_origin | is_term)).sum()))
    f, on_core = f[~both].copy(), on_core[~both]
    arr_none, dep_none = f.arrival.eq("None"), f.departure.eq("None")
    is_origin, is_term = f.seq.eq(0), f.seq.eq(f.last_seq)

    fill_arr = arr_none & ~is_origin
    fill_dep = dep_none & ~is_term
    log.add("none_fill", "one time missing away from origin/terminus: filled from the other time (zero halt assumed)",
            arrival_filled=int(fill_arr.sum()), departure_filled=int(fill_dep.sum()),
            of_which_core=int(((fill_arr | fill_dep) & on_core).sum()))
    f.loc[fill_arr, "arrival"] = f.loc[fill_arr, "departure"]
    f.loc[fill_dep, "departure"] = f.loc[fill_dep, "arrival"]

    f["arr_tod"], f["dep_tod"] = _tod_minutes(f.arrival), _tod_minutes(f.departure)
    base = (f.day - 1) * 1440
    f["day_offset"] = (f.day - 1).astype(int)
    f["arr_abs"], f["dep_abs"] = base + f.arr_tod, base + f.dep_tod
    # origin / terminus: the single time stands for both ends of the (zero) halt on the timeline
    f["arr_abs"] = f.arr_abs.fillna(f.dep_abs)
    f["dep_abs"] = f.dep_abs.fillna(f.arr_abs)

    roll = f.dep_abs < f.arr_abs
    log.add("midnight_halt", "halt crossing midnight (departure earlier than arrival): departure moved to next day",
            rows=int(roll.sum()), of_which_core=int((roll & f.station_code.isin(core_codes)).sum()))
    f.loc[roll, "dep_abs"] += 1440

    f = f.sort_values(["train_number", "seq"])
    bad_idx = []
    for _, g in f.groupby("train_number", sort=False):
        run = -1e18
        for idx, a, d in zip(g.index, g.arr_abs, g.dep_abs):
            if a < run:
                bad_idx.append(idx)
            else:
                run = d
    log.add("time_backwards", "arrival earlier than the previous kept departure: stop dropped",
            dropped=len(bad_idx), of_which_core=int(f.loc[bad_idx, "station_code"].isin(core_codes).sum()),
            trains=int(f.loc[bad_idx, "train_number"].nunique()))
    return f.drop(index=bad_idx)


# --------------------------------------------------------------------------- pipeline
def build_tables(raw_dir="data/raw"):
    """Run the whole Phase 2 pipeline. Returns (tables, log, extras)."""
    log = CleaningLog()
    sched, st, trains_meta = load_raw(raw_dir)
    log.add("raw", "raw schedules.json", rows=len(sched), trains=sched.train_number.nunique(),
            stations=sched.station_code.nunique())

    s, n_dup = drop_duplicate_stops(sched)
    log.add("dedup", "exact duplicate stop rows dropped (all columns equal except id)", dropped=n_dup, rows_after=len(s))
    s = add_sequence(s)

    ct = corridor_trains(s)
    counts = station_train_counts(corridor_slice(s, ct))
    core20 = counts[counts >= MIN_CORRIDOR_TRAINS]
    offline = [c for c in core20.index if c in OFFLINE_STATIONS]
    core_codes = [c for c in core20.index if c not in OFFLINE_STATIONS]
    log.add("corridor", f"corridor trains = stop at 2+ of {ANCHORS}", corridor_trains=len(ct),
            stop_at_4plus_anchors=int((ct.n_anchors >= 4).sum()), wide_stations=len(counts),
            stations_with_20plus_trains=len(core20), removed_offline=len(offline), core_stations=len(core_codes))

    full = s[s.train_number.isin(ct.index)]
    full_len = full.groupby("train_number").size().rename("n_full_route_stops")
    origin = full[full.seq == 0].set_index("train_number")
    dest = full[full.seq == full.last_seq].set_index("train_number")
    log.add("full_routes", "full routes of corridor trains", trains=len(ct), stops=len(full))

    timed = apply_time_rules(full, set(core_codes), log)
    t0 = timed.groupby("train_number").dep_abs.transform("first")
    timed = timed.assign(minutes_from_start=timed.arr_abs - t0, dep_min_from_start=timed.dep_abs - t0)
    timed["is_origin"] = timed.seq.eq(0)
    timed["is_terminus"] = timed.seq.eq(timed.last_seq)

    core_rows = timed[timed.station_code.isin(core_codes)].copy()
    log.add("core_filter", "stops kept at core stations", stops_before=len(timed), stops_after=len(core_rows),
            trains=core_rows.train_number.nunique())
    n_core = core_rows.groupby("train_number").size()
    short = n_core[n_core < MIN_CORE_STOPS].index
    never = sorted(set(ct.index) - set(n_core.index))
    log.add("short_trains", f"trains with fewer than {MIN_CORE_STOPS} core stops after filtering: dropped",
            dropped=len(short) + len(never), with_one_core_stop=len(short), with_zero_core_stops=len(never))
    core_rows = core_rows[~core_rows.train_number.isin(short)].sort_values(["train_number", "seq"])
    n_core = n_core.drop(short)
    core_rows["sequence"] = core_rows.groupby("train_number").cumcount()

    dup_visits = core_rows.duplicated(["train_number", "station_code"], keep=False)
    log.add("revisits", "train visits the same core station more than once (kept, distinct sequence)",
            rows=int(dup_visits.sum()), trains=int(core_rows[dup_visits].train_number.nunique()))

    first = core_rows.groupby("train_number").seq.min()
    last = core_rows.groupby("train_number").seq.max()
    full_last = core_rows.groupby("train_number").last_seq.first()
    entered = (first > 0).rename("entered_mid_route")
    exits = (last < full_last).rename("exits_mid_route")
    log.add("mid_route", "trains that join / leave the corridor away from their origin / terminus",
            trains=len(entered), entered_mid_route=int(entered.sum()), exits_mid_route=int(exits.sum()))

    core_rows["entered_mid_route"] = core_rows.train_number.map(entered)
    core_rows["halt_min"] = core_rows.dep_abs - core_rows.arr_abs
    arr_str = core_rows.arrival.where(~core_rows.is_origin)
    dep_str = core_rows.departure.where(~core_rows.is_terminus)
    stops = pd.DataFrame({
        "train": core_rows.train_number, "station": core_rows.station_code, "sequence": core_rows.sequence,
        "scheduled_arrival": arr_str.where(arr_str.ne("None")),
        "scheduled_departure": dep_str.where(dep_str.ne("None")),
        "day_offset": core_rows.day_offset, "halt_min": core_rows.halt_min,
        "minutes_from_start": core_rows.minutes_from_start, "dep_min_from_start": core_rows.dep_min_from_start,
        "is_halt": (core_rows.halt_min > 0) | core_rows.is_origin | core_rows.is_terminus,
        "entered_mid_route": core_rows.entered_mid_route, "original_seq": core_rows.seq,
    }).reset_index(drop=True)

    meta = trains_meta.drop_duplicates("number").set_index("number")
    names = full.drop_duplicates("train_number").set_index("train_number").train_name
    trains = pd.DataFrame({"number": n_core.index})
    trains["name"] = trains.number.map(names)
    trains["type"] = [classify_train(n, nm) for n, nm in zip(trains.number, trains.name)]
    trains["is_slip"] = trains.name.str.contains("Slip", case=False) | trains.number.str.contains("Slip", case=False)
    trains["source"] = trains.number.map(origin.station_code)
    trains["destination"] = trains.number.map(dest.station_code)
    trains["runs_on"] = "daily - assumed"
    trains["entered_mid_route"] = trains.number.map(entered)
    trains["exits_mid_route"] = trains.number.map(exits)
    trains["n_full_route_stops"] = trains.number.map(full_len)
    trains["n_core_stops"] = trains.number.map(n_core)
    trains["trimmed_stops"] = trains.n_full_route_stops - trains.n_core_stops
    trains["full_route_distance_km"] = trains.number.map(meta.distance)
    # clock times on the journey timeline (minutes from 00:00 of journey day 1), used for rake reuse
    trains["origin_departure_min"] = trains.number.map(timed.groupby("train_number").dep_abs.first())
    trains["terminus_arrival_min"] = trains.number.map(timed.groupby("train_number").arr_abs.last())
    log.add("types", "train type counts (rules in clean.TYPE_RULES)", **trains.type.value_counts().to_dict())
    log.add("slips", "coach-slip entries (name contains 'Slip'): duplicate movement of another train",
            trains=int(trains.is_slip.sum()))

    nm = s.drop_duplicates("station_code").set_index("station_code").station_name
    stn = st.drop_duplicates("code").set_index("code")
    stations = pd.DataFrame({"code": core_codes})
    stations["name"] = stations.code.map(nm)
    stations["display_name"] = [DISPLAY_NAMES.get(c, n) for c, n in zip(stations.code, stations.name)]
    stations["is_junction"] = [is_junction_guess(n, c) for n, c in zip(stations.name, stations.code)]
    stations["lat"] = stations.code.map(stn.lat)
    stations["lon"] = stations.code.map(stn.lon)
    stations["zone"] = stations.code.map(stn.zone)
    stations["n_corridor_trains"] = stations.code.map(core20)

    offline_df = pd.DataFrame({"code": offline})
    offline_df["name"] = offline_df.code.map(nm)
    offline_df["n_corridor_trains"] = offline_df.code.map(core20)
    offline_df["reason"] = offline_df.code.map(OFFLINE_STATIONS)
    offline_df["lat"] = offline_df.code.map(stn.lat)
    offline_df["lon"] = offline_df.code.map(stn.lon)

    wide_df = pd.DataFrame({"code": counts.index, "name": counts.index.map(nm), "n_corridor_trains": counts.values,
                            "lat": counts.index.map(stn.lat), "lon": counts.index.map(stn.lon)})
    wide_df = wide_df.sort_values("n_corridor_trains", ascending=False)

    seg = segment_diagnostics(stops, stations)
    trains["max_gap_km"] = trains.number.map(seg.groupby("train").km.max())
    trains["leaves_main_line"] = trains.max_gap_km > OFF_LINE_HOP_KM
    log.add("off_line_hops", f"trains with two consecutive core stops more than {OFF_LINE_HOP_KM} km apart (straight line): "
            "they skip the main line between those stops (flag `leaves_main_line`)",
            trains=int(trains.leaves_main_line.sum()), hops=int((seg.km > OFF_LINE_HOP_KM).sum()))

    chain, branch, bad = main_line_chain(core_rows)
    log.add("chain", "main line as one NDLS->MGS chain; branch/loop stops and trains that reverse or loop",
            chain_stations=len(chain), branch_stops=len(branch), inconsistent_trains=len(bad))
    extras = {"chain": chain, "branch_stops": branch, "chain_inconsistent_trains": bad, "offline": offline_df,
              "wide": wide_df, "timed_full": timed, "raw_corridor_trains": ct, "dedup_schedule": s}
    return {"stations": stations, "trains": trains, "stops": stops}, log, extras


# --------------------------------------------------------------------------- diagnostics
def segment_diagnostics(stops, stations):
    """Consecutive-stop segments with straight-line km (haversine on lat/lon), minutes and km/h.

    Straight-line distance understates track distance, and times are whole minutes, so short
    segments give unreliable speeds. Used for diagnostics only.
    """
    import numpy as np

    s = stops.sort_values(["train", "sequence"]).merge(
        stations[["code", "lat", "lon"]], left_on="station", right_on="code")
    s = s.sort_values(["train", "sequence"])
    g = s.groupby("train")
    s["from_station"], s["lat0"], s["lon0"] = g.station.shift(), g.lat.shift(), g.lon.shift()
    s["dep0"] = g.dep_min_from_start.shift()
    la0, lo0, la1, lo1 = (np.radians(s[c]) for c in ("lat0", "lon0", "lat", "lon"))
    a = np.sin((la1 - la0) / 2) ** 2 + np.cos(la0) * np.cos(la1) * np.sin((lo1 - lo0) / 2) ** 2
    s["km"] = 2 * 6371 * np.arcsin(np.sqrt(a))
    s["minutes"] = s.minutes_from_start - s.dep0
    s["kmh"] = s.km / (s.minutes / 60)
    return s.dropna(subset=["km"])[["train", "from_station", "station", "km", "minutes", "kmh"]]


# --------------------------------------------------------------------------- outputs
def write_outputs(tables, log, extras, out_dir="data/processed"):
    """Write parquet + csv copies of the three tables and the station lists. Never touches data/raw."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        df.to_csv(out / f"{name}.csv", index=False)
    tables["stations"].sort_values("n_corridor_trains", ascending=False).to_csv(
        out / "corridor_stations_core.csv", index=False)
    extras["wide"].to_csv(out / "corridor_stations_wide.csv", index=False)
    extras["offline"].to_csv(out / "corridor_stations_offline.csv", index=False)


def segment_counts(chain, anchors=ANCHORS):
    """Number of core stations strictly between / at consecutive anchors along the chain."""
    pos = {c: chain.index(c) for c in anchors if c in chain}
    ordered = sorted(pos, key=pos.get)
    out = []
    for a, b in zip(ordered, ordered[1:]):
        out.append((f"{a} - {b}", pos[b] - pos[a] + 1))
    return out


def render_log_markdown(log, tables, extras):
    """Markdown text for reports/phase2_cleaning_log.md."""
    L = log.entries
    g = {e["step"]: e["counts"] for e in L}
    seg = segment_diagnostics(tables["stops"], tables["stations"])
    v = seg[seg.minutes > 0]
    lines = ["# Phase 2 cleaning log: Corridor B", "",
             "Generated by `python -m src.data.clean` (functions in `src/data/clean.py`). Raw files in `data/raw/` are never modified.", "",
             "## Assumptions", "",
             "- **Every train runs daily.** The raw file has no day-of-week information (weekly/bi-weekly trains are treated as daily).",
             "- `day` is the 1-based journey day of the row's times. A halt crossing midnight has one `day`, so its departure is rolled to the next day.",
             "- Row order within a train is the raw `id` order (consecutive within each train).",
             "- A row with only one time (away from origin/terminus) gets the other time (zero halt). Rows with neither time are dropped. Nothing else is invented.",
             "- Anchors: " + ", ".join(ANCHORS) + " (Old Delhi dropped). A corridor train has a schedule row at 2+ anchors. "
             "**The raw schedule lists pass-through stations with arrival = departure**, so this is 'passes or stops', not 'halts at'. "
             "Counting only real halts (halt > 0) gives 217 of the 229 trains.", "",
             "## Steps and counts", "", "| Step | Rule | Counts |", "|---|---|---|"]
    for e in L:
        c = ", ".join(f"{k} = {v_}" for k, v_ in e["counts"].items())
        lines.append(f"| {e['step']} | {e['message']} | {c} |")
    lines += ["", "## Core stations", "",
              f"- {g['corridor']['stations_with_20plus_trains']} stations are served by 20+ corridor trains; "
              f"{g['corridor']['removed_offline']} of them are off-line and were removed "
              f"(see `data/processed/corridor_stations_offline.csv`), leaving **{g['corridor']['core_stations']}** core stations.",
              "- The earlier note of 24 off-line stations was my miscount: only 15 were ever listed. Three (DLI, DSA, VVB) fell out of the 20+ "
              "set by themselves once DLI stopped being an anchor, so 12 were removed explicitly.",
              "- Stations per segment between anchors (inclusive of both anchors): "
              + "; ".join(f"{k}: {n}" for k, n in segment_counts(extras["chain"])) + ".",
              f"- Main-line chain (NDLS to MGS): {len(extras['chain'])} stations, branch/loop stops not on the chain: "
              f"{extras['branch_stops'] or 'none'}.",
              f"- {len(extras['chain_inconsistent_trains'])} trains visit chain stations out of chain order. All are around Prayagraj "
              "(ALD, NYN, COI): the stop sequences (e.g. NYN ALD NYN, or NYN ALD COI) suggest trains reversing at or looping around Prayagraj Jn, "
              "so ALD-NYN-COI looks like a triangle rather than a straight chain (inferred from the sequences, not checked on a map). Trains: " + ", ".join(extras["chain_inconsistent_trains"]) + ".", "",
              "## Train types", "", "Rules (first match wins):", "", "```", TYPE_RULES, "```", "",
              "Types come from the name and number only, not from `trains.json`. Coach-slip entries (name or number contains 'Slip') "
              "are flagged `is_slip`; they duplicate the movement of another train.", "",
              "## What may trouble the simulator", "",
              f"- **Pass-through rows.** Only {int(tables['stops'].is_halt.sum())} of {len(tables['stops'])} core stop rows are real halts "
              "(`is_halt`); the rest are pass times. Halts are whole minutes, mostly 0.",
              f"- **Whole-minute times vs short segments.** {int((seg.minutes <= 0).sum())} consecutive-stop segments take 0 minutes and "
              f"{int(((seg.minutes > 0) & (seg.minutes <= 1)).sum())} take 1 minute. Straight-line speed on the other "
              f"{len(v)} has median {v.kmh.median():.0f} km/h; {int((v.kmh > 130).sum())} exceed 130 km/h (max {v.kmh.max():.0f}, "
              "from 1-minute segments of a few km) and "
              f"{int((v.kmh < 10).sum())} are under 10 km/h (e.g. KNS-PJY, 0.12 km apart, 16-22 min; cause not investigated). "
              "Do not use per-segment speed from this data; use whole-section times or add track distances.",
              f"- **Trains that skip the main line.** {g['off_line_hops']['trains']} trains have two consecutive core stops more than "
              f"{OFF_LINE_HOP_KM} km apart ({g['off_line_hops']['hops']} hops; longest {seg.minutes.max():.0f} min, "
              "e.g. 14003/14004 Delhi-Farakka jump Ghaziabad to Mughalsarai, 677 km). They touch two anchors but run between them on a "
              "different line (for example via Lucknow-Varanasi, or Prayagraj to Mughalsarai by another route). Flag `leaves_main_line` "
              "in `trains`; a simulator that moves them along the main line chain would be wrong for these.",
              f"- **Revisits.** {g['revisits']['trains']} trains visit a core station twice (Prayagraj reversal); sequence keeps both visits.",
              f"- **Mid-route entry/exit.** {g['mid_route']['entered_mid_route']} of {g['mid_route']['trains']} trains enter the corridor "
              f"away from their origin and {g['mid_route']['exits_mid_route']} leave before their terminus, so their incoming delay is unknown.",
              "- **No per-stop distance** in the raw file; `segment_diagnostics` uses straight-line distance from station coordinates.",
              "- **Daily assumption** inflates traffic: weekly/bi-weekly trains run every day in the simulation.",
              "- **Old timetable** (pre-2018 names/codes, no recent trains). Freight is absent.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    t, lg, ex = build_tables()
    write_outputs(t, lg, ex)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/phase2_cleaning_log.md").write_text(render_log_markdown(lg, t, ex), encoding="utf-8")
    print({k: len(v) for k, v in t.items()})
