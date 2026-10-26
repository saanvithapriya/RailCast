"""Run the Corridor B delay simulator (SIMULATED data).

    python -m src.simulation.run --start 2025-01-01 --end 2025-01-31 --out data/processed/sim_pilot_jan
"""
import argparse
import datetime
import random
import time

from .config import load_config
from .delays import DelayModel
from .engine import MoveParams, Simulator
from .ledger import Ledger
from .network import load_specs, scheduled_pair_gaps
from .recorder import Recorder, write_outputs
from .weather import EPOCH, SyntheticWeather


def to_day(s):
    return (datetime.date.fromisoformat(s) - EPOCH).days


def move_params(cfg):
    mv, rk = cfg["movement"], cfg["rake"]
    return MoveParams(exit_gap=mv["exit_gap_min"], yield_window=mv["yield_window_min"],
                      max_yield_hold=mv["max_yield_hold_min"], yield_min_conflict=mv["yield_min_conflict_min"], overtake_mode=mv["overtake_mode"],
                      loop_capacity=int(mv["loop_capacity"]), overtake_penalty=mv["overtake_penalty_min"], lookback=int(mv["yield_lookback_sections"]),
                      min_turnaround=rk["min_turnaround_min"], offcorridor_recovery=rk["offcorridor_recovery"],
                      rake_noise_sd=rk["noise_sd_min"])


def simulate(start_day, end_day, cfg=None, built=None, weather=None, out_dir=None, progress=False, log_queues=False):
    """Simulate calendar days start_day..end_day (day 0 = 2025-01-01) after a warm-up.

    Returns a dict with the recorder, ledger, delay model, specs, link stats and timing.
    """
    t_start = time.time()
    cfg = cfg or load_config()
    net, specs, link_stats, _ = built or load_specs(cfg=cfg)
    seed = cfg["simulation"]["seed"]
    first = start_day - int(cfg["simulation"]["warmup_days"])
    # the weather chain starts 120 days before the first simulated day; the 2025 default anchor (-120) is kept
    # so existing 2025 runs are unchanged, and runs before 2025 move the anchor back
    anchor = SyntheticWeather.ANCHOR if first >= SyntheticWeather.ANCHOR else first - 120
    weather = weather or SyntheticWeather(cfg, seed, anchor=anchor)
    dm = DelayModel(net, cfg, weather, seed)
    rec = Recorder(specs, net, start_day, end_day)
    led = Ledger(rec)

    def finish(run):
        rec.finish(run)
        led.finish(run)

    sim = Simulator(net, move_params(cfg), dm, led, finish)
    if cfg["movement"].get("assume_scheduled_pairs_feasible", True):
        sim.pairs = scheduled_pair_gaps(specs, net, cfg["movement"]["exit_gap_min"])
    if log_queues:
        sim.queue_log = []
    order = sorted(specs)
    for d in range(first, end_day + 1):
        dm.generate_day(d)
        for tn in order:
            sim.add_run(specs[tn], d, random.Random(f"{seed}-{d}-{tn}"))
        sim.run_until((d + 1) * 1440.0)
        sim.forget_before(d - 6)
        if progress and (d - first) % 10 == 0:
            print(f"  day {d - first + 1}/{end_day - first + 1}  {time.time() - t_start:.0f}s", flush=True)
    sim.drain()
    res = {"queue_log": sim.queue_log, "recorder": rec, "ledger": led, "delays": dm, "specs": specs, "net": net, "links": link_stats,
           "weather": weather, "cfg": cfg, "sim_seconds": time.time() - t_start, "start_day": start_day,
           "end_day": end_day}
    if out_dir:
        t = time.time()
        res["sizes"] = write_outputs(rec, dm, weather, cfg, out_dir, start_day, end_day,
                                     {"start": str(EPOCH + datetime.timedelta(days=start_day)),
                                      "end": str(EPOCH + datetime.timedelta(days=end_day)), "seed": seed})
        res["write_seconds"] = time.time() - t
    return res


def main():
    ap = argparse.ArgumentParser(description="Corridor B delay simulator (SIMULATED data)")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-01-31")
    ap.add_argument("--out", default="data/processed/sim")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    res = simulate(to_day(args.start), to_day(args.end), load_config(args.config), out_dir=args.out, progress=True)
    print(f"simulated {res['recorder'].run_count} train runs in {res['sim_seconds']:.0f}s "
          f"(+{res['write_seconds']:.0f}s writing); ledger max mismatch {res['ledger'].max_mismatch:.2e}")


if __name__ == "__main__":
    main()
