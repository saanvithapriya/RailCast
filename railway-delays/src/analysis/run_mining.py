"""Run the frozen mining pipeline on every window and scenario needed for the Phase 5 notebooks (SIMULATED data).

    python -m src.analysis.run_mining

Windows: the frozen application window (2024-11-01..2025-08-31, medium) and 2025-01-01..2025-08-31 for medium (ground-truth
validation window and robustness baseline), low and high. Also stores the cascade sensitivity table (P and G) for the
decision window and the application window. The decision window itself was run when the method was designed.
"""
import time
from pathlib import Path

import pandas as pd

from src.analysis import cascades as cs
from src.analysis import decomposition as dc
from src.analysis import eda, passages
from src.analysis import mining_pipeline as mp
from src.analysis.calibrate import load_frozen
from src.graph.station_graph import build_station_tables
from src.graph.timetable import load_timetable

RUNS = [("medium", "2024-11-01", "2025-08-31"), ("medium", "2025-01-01", "2025-08-31"),
        ("low", "2025-01-01", "2025-08-31"), ("high", "2025-01-01", "2025-08-31")]
SENS = [("medium", "2024-01-01", "2024-10-31"), ("medium", "2024-11-01", "2025-08-31")]


def sensitivity_for(scenario, start, end, tt, nodes):
    prm = load_frozen()
    df, tt, nodes, bounds = eda.load_window(start, end, scenario=scenario, tt=tt, nodes=nodes)
    p = passages.attach_leaders(passages.build_passages(df, bounds))
    runs = passages.build_runs(df, tt, bounds)
    del df
    p = dc.decompose(p, prm.slack_fraction)
    s = cs.sensitivity(p, runs, prm)
    s.insert(0, "window", f"{start}..{end}")
    s.insert(0, "scenario", scenario)
    return s


def main():
    tt = load_timetable()
    nodes, _, _ = build_station_tables(tt)
    for scenario, a, b in RUNS:
        t = time.time()
        mp.run_pipeline(a, b, scenario=scenario, tt=tt)
        print(f"{scenario} {a}..{b} finished in {time.time() - t:.0f}s", flush=True)
    parts = []
    for scenario, a, b in SENS:
        t = time.time()
        parts.append(sensitivity_for(scenario, a, b, tt, nodes))
        print(f"sensitivity {scenario} {a}..{b} in {time.time() - t:.0f}s", flush=True)
    out = Path("data/processed/mining")
    pd.concat(parts, ignore_index=True).to_parquet(out / "sensitivity.parquet", index=False)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
