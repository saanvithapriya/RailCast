"""Low / medium / high congestion scenarios, full calendar 2025 (SIMULATED data).

    python -m src.simulation.scenarios

Scenarios differ only in headway and freight rate (set in advance, not tuned); seed, weather and every
other parameter are the defaults in config.yaml, so differences between them come from congestion alone.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import causes as C
from .config import load_config
from .network import load_specs
from .run import simulate, to_day

SCENARIOS = {
    "low": {"movement": {"headway_min": 3}, "freight": {"rate_per_section_hour": 0.02}},
    "medium": {"movement": {"headway_min": 4}, "freight": {"rate_per_section_hour": 0.04}},
    "high": {"movement": {"headway_min": 6}, "freight": {"rate_per_section_hour": 0.06}},
}


def max_queue(wait_start, wait_end):
    ev = np.r_[wait_start, wait_end]
    w = np.r_[np.ones(len(wait_start)), -np.ones(len(wait_start))]
    o = np.lexsort((w, ev))              # at equal time, departures before arrivals
    return int(np.cumsum(w[o]).max())


def run_scenario(name, overrides, root, start="2025-01-01", end="2025-12-31"):
    cfg = load_config(overrides=overrides)
    built = load_specs(cfg=cfg)
    out = Path(root) / name
    t = time.time()
    res = simulate(to_day(start), to_day(end), cfg, built=built, out_dir=out, log_queues=True)
    rec = res["recorder"]
    # queues: largest simultaneous queue per section-direction over the year
    q = pd.DataFrame(res["queue_log"], columns=["section", "direction", "wait_start", "wait_end", "train"])
    q = q[q.wait_end - q.wait_start > 0.5]
    biggest = max((max_queue(g.wait_start.values, g.wait_end.values) for _, g in q.groupby(["section", "direction"])), default=0)
    waits_3plus = sum(1 for _, g in q.groupby(["section", "direction"]) if max_queue(g.wait_start.values, g.wait_end.values) >= 3)
    q["is_simulated"] = True
    q.to_parquet(out / "queues.parquet", index=False)
    # metrics and the ledger check, month by month
    n = n_ok = 0
    arr = []
    added = {}
    check_rows = check_match = 0
    check_max_err = 0.0
    for m in rec.months():
        df, dfa = rec.tables(m)
        a = df.arrival_delay.dropna().to_numpy()
        arr.append(a)
        pos = dfa[(dfa.phase == "arr") & (dfa.cause_code > 0)]
        for code, v in pos.groupby("cause_code").new_minutes.sum().items():
            added[int(code)] = added.get(int(code), 0.0) + float(v)
        comp = pos.groupby("obs_id").minutes.sum()
        late = df[df.arrival_delay >= 0.5].set_index("obs_id")
        both = late.join(comp, how="inner")
        check_rows += len(both)
        err = (both.arrival_delay.astype(float) - both.minutes.astype(float)).abs()
        check_match += int((err <= 0.05).sum())
        check_max_err = max(check_max_err, float(err.max()) if len(err) else 0.0)
    allv = np.concatenate(arr)
    tot_added = sum(added.values())
    others = sum(added.get(c, 0.0) for c in (C.SECTION_WAIT, C.YIELD, C.RAKE))
    total_bytes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    summary = {
        "headway_min": cfg["movement"]["headway_min"], "freight_rate_per_section_hour": cfg["freight"]["rate_per_section_hour"],
        "lookback": cfg["movement"]["yield_lookback_sections"],
        "observation_rows": int(len(allv)), "train_runs": rec.run_count,
        "share_within_10min": float((allv <= 10).mean()), "mean_delay_min": float(allv.mean()),
        "p99_delay_min": float(np.quantile(allv, 0.99)), "share_added_by_other_trains": others / tot_added,
        "share_added_weather": added.get(C.WEATHER, 0.0) / tot_added,
        "largest_queue_trains": biggest, "section_directions_with_queue_3plus": waits_3plus,
        "runtime_seconds": round(time.time() - t, 1), "simulate_seconds": round(res["sim_seconds"], 1),
        "size_mb": round(total_bytes / 1e6, 1),
        "ledger_max_mismatch_min": res["ledger"].max_mismatch,
        "attribution_check_rows": check_rows, "attribution_check_share_within_0.05min": check_match / max(check_rows, 1),
        "attribution_check_max_error_min": check_max_err,
    }
    (out / "scenario_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_medium_2024(root="data/processed/sim"):
    """Medium scenario for calendar 2024 with a different seed (2024), written to <root>/medium_2024/.

    Same configuration as the 2025 medium scenario otherwise, so weather and incidents differ but the
    timetable, headway and freight rate are the same. SIMULATED.
    """
    ov = {**SCENARIOS["medium"], "simulation": {"seed": 2024}}
    return run_scenario("medium_2024", ov, root, start="2024-01-01", end="2024-12-31")


def main(root="data/processed/sim"):
    summaries = {}
    for name, ov in SCENARIOS.items():
        summaries[name] = run_scenario(name, ov, root)
        print(name, json.dumps(summaries[name]), flush=True)
    Path(root, "scenarios_summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    import sys

    if "--medium-2024" in sys.argv:
        s = run_medium_2024()
        print("medium_2024", json.dumps(s), flush=True)
    else:
        main()
