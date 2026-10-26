"""Pilot run: one winter month (January) and one summer month (June) of 2025. SIMULATED data.

    python -m src.simulation.pilot
"""
import json
import time
from pathlib import Path

import pandas as pd

from .config import load_config
from .network import load_specs
from .run import simulate, to_day

MONTHS = {"jan": ("2025-01-01", "2025-01-31"), "jun": ("2025-06-01", "2025-06-30")}


def main(out_root="data/processed/sim_pilot"):
    cfg = load_config()
    built = load_specs(cfg=cfg)
    timing = {}
    for name, (a, b) in MONTHS.items():
        out = Path(out_root) / name
        t = time.time()
        res = simulate(to_day(a), to_day(b), cfg, built=built, out_dir=out, log_queues=True, progress=False)
        q = pd.DataFrame(res["queue_log"], columns=["section", "direction", "wait_start", "wait_end", "train"])
        q = q[(q.wait_end >= to_day(a) * 1440.0) & (q.wait_start < (to_day(b) + 1) * 1440.0)]
        q["is_simulated"] = True
        q.to_parquet(out / "queues.parquet", index=False)
        total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        timing[name] = {"runs": res["recorder"].run_count, "simulate_seconds": round(res["sim_seconds"], 1),
                        "write_seconds": round(res["write_seconds"], 1), "total_seconds": round(time.time() - t, 1),
                        "bytes": total, "ledger_max_mismatch": res["ledger"].max_mismatch,
                        "observation_rows": sum(len(b) for b in res["recorder"].obs.values())}
        print(name, timing[name], flush=True)
    (Path(out_root) / "timing.json").write_text(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
