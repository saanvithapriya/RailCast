"""Calibration of the frozen mining parameters on the DECISION window only (2024-01-01 .. 2024-10-31). SIMULATED data.

Writes configs/mining_frozen.yaml (slack fractions per train type, gap window G, and the fixed thresholds). The rest of
the pipeline loads that file and never re-estimates anything. Run once:  python -m src.analysis.calibrate
"""
import json
import time
from pathlib import Path

import yaml

from src import splits
from src.analysis import decomposition as dc
from src.analysis import eda, passages

FROZEN = Path(__file__).resolve().parent.parent.parent / "configs" / "mining_frozen.yaml"
FIXED = {"L_late": 5.0, "A_link": 2.0, "P": 15.0, "T_entry": 15.0, "R_min": 5.0}     # set before looking at results


def load_frozen():
    """The frozen parameters as a decomposition.Params."""
    c = yaml.safe_load(FROZEN.read_text())
    return dc.Params(G=float(c["G"]), L_late=float(c["L_late"]), A_link=float(c["A_link"]), P=float(c["P"]),
                     T_entry=float(c["T_entry"]), R_min=float(c["R_min"]), G_rake=float(c["G_rake"]),
                     slack_fraction=tuple(sorted((k, float(v)) for k, v in c["slack_fraction"].items())))


def choose_rake_window(runs, tt, edges=(0, 60, 90, 120, 180, 240, 360, 600, 1500), min_corr=0.30, min_pairs=500):
    """Scheduled turnaround limit G_rake. Runs whose rake feeder is known are binned by the scheduled turnaround; in each
    bin the correlation between the run entry delay and the feeder exit delay is computed. Bins with fewer than
    `min_pairs` pairs are too noisy to judge and are skipped. G_rake is the upper edge of the last bin before the first
    judged bin whose correlation falls below `min_corr`. Decision window only."""
    import numpy as np
    import pandas as pd

    r = runs.reset_index()
    idx = pd.Series(np.arange(len(r)), index=pd.MultiIndex.from_arrays([r.train.to_numpy(), r.day.to_numpy()]))
    f = idx.reindex(pd.MultiIndex.from_arrays([r.feeder_train.fillna("").to_numpy(), (r.day - r.feeder_k).to_numpy()])).to_numpy()
    ok = ~np.isnan(f) & r.from_start.to_numpy() & r.feeder_gap.notna().to_numpy()
    d = r[ok].assign(f_exit=r.exit_delay.to_numpy()[f[ok].astype(int)])
    curve, G = [], float(edges[-1])
    stopped = False
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = d[(d.feeder_gap > lo) & (d.feeder_gap <= hi)]
        c = float(s.entry_delay.corr(s.f_exit)) if len(s) > 2 else float("nan")
        curve.append({"gap_from": lo, "gap_to": hi, "pairs": int(len(s)), "corr_entry_vs_feeder_exit": c})
        if not stopped and len(s) >= min_pairs and c < min_corr:
            G, stopped = float(lo), True
    return G, curve


def calibrate():
    a, b = splits.eda_windows()["decision"]
    t = time.time()
    df, tt, nodes, bounds = eda.load_window(a, b)
    p = passages.attach_leaders(passages.build_passages(df, bounds))
    frac, n_free = dc.estimate_slack_fractions(p)
    p = dc.decompose(p, tuple(frac.items()))
    curve = dc.gap_response_curve(p, FIXED["L_late"])
    G = dc.choose_gap_window(curve)
    G_rake, rake_curve = choose_rake_window(passages.build_runs(df, tt, bounds), tt)
    out = {"window": [a, b], "source": "decision window only", **FIXED, "G": G, "G_rake": G_rake,
           "slack_fraction": {k: round(v, 4) for k, v in frac.items()}, "slack_fit_passages": n_free}
    FROZEN.write_text("# FROZEN mining parameters: calibrated on 2024-01-01..2024-10-31 only (SIMULATED data); never re-estimated.\n"
                      + yaml.safe_dump(out, sort_keys=False))
    Path("data/processed/mining").mkdir(parents=True, exist_ok=True)
    Path("data/processed/mining/calibration_decision_window.json").write_text(json.dumps({"gap_curve": curve.to_dict("records"), "rake_curve": rake_curve, **out}, indent=2, default=float))
    print(f"calibrated in {time.time() - t:.0f}s on {len(p):,} passages:", out)
    print(curve.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    calibrate()
