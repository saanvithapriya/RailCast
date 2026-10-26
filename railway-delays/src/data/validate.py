"""Validation checks for the cleaned Corridor B tables. Exits non-zero on any failure.

Usage:  python -m src.data.validate [processed_dir]
"""
import sys
from pathlib import Path

import pandas as pd


def validate(stations, trains, stops):
    """Return a list of (check_name, n_failures, detail) for every failed check."""
    failures = []

    def check(name, bad):
        if len(bad):
            failures.append((name, len(bad), bad.head(5).to_string() if hasattr(bad, "to_string") else str(bad)))

    # every stop's station exists in stations, every stop's train exists in trains
    check("stop station not in stations", stops[~stops.station.isin(stations.code)])
    check("stop train not in trains", stops[~stops.train.isin(trains.number)])
    check("duplicate station code in stations", stations[stations.code.duplicated()])
    check("duplicate train number in trains", trains[trains.number.duplicated()])

    # no train has a single stop
    n = stops.groupby("train").size()
    check("train with fewer than 2 stops", n[n < 2].to_frame("stops"))
    check("train without stops", trains[~trains.number.isin(stops.train)])

    # no duplicate (train, station, sequence); sequence runs 0..n-1 in order
    check("duplicate (train, station, sequence)", stops[stops.duplicated(["train", "station", "sequence"], keep=False)])
    s = stops.sort_values(["train", "sequence"])
    expected = s.groupby("train").cumcount()
    check("sequence not contiguous from 0", s[s.sequence != expected])

    # no negative halts
    check("negative halt", stops[stops.halt_min < 0])

    # times monotonic along each route: departure >= arrival, next arrival >= previous departure
    check("departure before arrival", s[s.dep_min_from_start < s.minutes_from_start])
    prev_dep = s.groupby("train").dep_min_from_start.shift()
    check("arrival before previous departure", s[s.minutes_from_start < prev_dep])
    return failures


def check_unchanged(baseline_dir, new_dir):
    """Confirm every table keeps its old rows and old columns unchanged (new columns are allowed).

    Returns a list of problems (empty if the new tables only add columns).
    """
    problems = []
    for name in ("stations", "trains", "stops"):
        old = pd.read_parquet(Path(baseline_dir) / f"{name}.parquet")
        new = pd.read_parquet(Path(new_dir) / f"{name}.parquet")
        if len(old) != len(new):
            problems.append(f"{name}: row count {len(old)} -> {len(new)}")
            continue
        missing = [c for c in old.columns if c not in new.columns]
        if missing:
            problems.append(f"{name}: columns removed {missing}")
        same = [c for c in old.columns if c in new.columns]
        if not old[same].equals(new[same]):
            diff = [c for c in same if not old[c].equals(new[c])]
            problems.append(f"{name}: values changed in {diff}")
        added = [c for c in new.columns if c not in old.columns]
        print(f"{name}: {len(new)} rows unchanged; old columns identical; added columns {added}")
    return problems


def main(processed_dir="data/processed"):
    p = Path(processed_dir)
    stations, trains, stops = (pd.read_parquet(p / f"{n}.parquet") for n in ("stations", "trains", "stops"))
    failures = validate(stations, trains, stops)
    print(f"checked {len(stations)} stations, {len(trains)} trains, {len(stops)} stops")
    if failures:
        for name, n, detail in failures:
            print(f"FAIL {name}: {n}\n{detail}")
        sys.exit(1)
    print("all validation checks passed")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--baseline":
        issues = check_unchanged(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "data/processed")
        if issues:
            print("CHANGED:", *issues, sep="\n  ")
            sys.exit(1)
        print("existing columns and counts are unchanged")
    else:
        main(*sys.argv[1:])
