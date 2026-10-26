"""One-off comparison of yield_lookback_sections = 1 vs 2 on the January and June pilots (SIMULATED).

Decision rule fixed before looking at the results:
* Shatabdi is "fixed" if, in clear weather, both its mean arrival delay and its delay added by other
  trains per 100 rows are <= Superfast's.
* Another train type is "worse" if its clear-weather mean delay rises by more than 1.0 minute, or the
  overall share of arrivals within 10 minutes falls by more than 1 percentage point.
"""
import json
from pathlib import Path

import pandas as pd

from . import causes as C
from .config import load_config
from .network import load_specs
from .run import simulate, to_day
from .weather import day_to_date

MONTHS = {"jan": ("2025-01-01", "2025-01-31"), "jun": ("2025-06-01", "2025-06-30")}
ORDER = ["Rajdhani", "Duronto", "Shatabdi", "Superfast", "Mail/Express", "Passenger", "other"]


def metrics(lookback):
    cfg = load_config(overrides={"movement": {"yield_lookback_sections": lookback}})
    built = load_specs(cfg=cfg)
    frames, attrs, wx = [], [], []
    for name, (a, b) in MONTHS.items():
        res = simulate(to_day(a), to_day(b), cfg, built=built)
        rec = res["recorder"]
        df, dfa = rec.tables(rec.months()[0])
        df["train"] = df.train.astype(str)
        df["type"] = df.train.map({t: s.type for t, s in res["specs"].items()})
        df["run"] = name
        dfa["run"] = name
        w = pd.DataFrame({"date": [pd.Timestamp(day_to_date(d)) for d in range(to_day(a), to_day(b) + 1)],
                          "state": [res["weather"].state(d) for d in range(to_day(a), to_day(b) + 1)]})
        frames.append(df.merge(w, on="date", how="left"))
        attrs.append(dfa)
    df, dfa = pd.concat(frames, ignore_index=True), pd.concat(attrs, ignore_index=True)
    d = df.dropna(subset=["arrival_delay"])
    clear = d[d.state == "clear"]
    out = pd.DataFrame({
        "mean delay (all)": d.groupby("type").arrival_delay.mean(),
        "mean delay (clear)": clear.groupby("type").arrival_delay.mean(),
        "share <=10 (all)": d.groupby("type").arrival_delay.apply(lambda s: (s <= 10).mean()),
    }).reindex(ORDER)
    ids = clear[["run", "obs_id", "type"]]
    ad = dfa[(dfa.phase == "arr") & dfa.cause_code.isin([C.SECTION_WAIT, C.YIELD, C.RAKE])].merge(ids, on=["run", "obs_id"])
    out["added by others / 100 rows (clear)"] = ad.groupby("type").new_minutes.sum() / clear.groupby("type").size() * 100
    ch = clear.groupby(["run", "train", "date"]).arrival_delay.last()
    overall = {"share <=10 (all)": float((d.arrival_delay <= 10).mean()), "mean delay (all)": float(d.arrival_delay.mean()),
               "p99": float(d.arrival_delay.quantile(.99))}
    sh = clear[clear.type == "Shatabdi"].groupby("train").arrival_delay.mean()
    return out, overall, sh


def main():
    res = {}
    for lb in (1, 2):
        out, overall, sh = metrics(lb)
        res[lb] = (out, overall, sh)
        print(f"\n=== lookback {lb}")
        print(out.round(2).to_string())
        print({k: round(v, 3) for k, v in overall.items()})
        print("Shatabdi trains, clear mean:", sh.round(1).to_dict(), flush=True)
    o1, a1, _ = res[1]
    o2, a2, _ = res[2]
    sup1, sup2 = o2.loc["Superfast"], o2.loc["Superfast"]
    shat = o2.loc["Shatabdi"]
    fixed = (shat["mean delay (clear)"] <= sup2["mean delay (clear)"]
             and shat["added by others / 100 rows (clear)"] <= sup2["added by others / 100 rows (clear)"])
    worse = {t: float(o2.loc[t, "mean delay (clear)"] - o1.loc[t, "mean delay (clear)"]) for t in ORDER
             if t != "Shatabdi" and o2.loc[t, "mean delay (clear)"] - o1.loc[t, "mean delay (clear)"] > 1.0}
    share_drop = a1["share <=10 (all)"] - a2["share <=10 (all)"]
    verdict = {"shatabdi_fixed": bool(fixed), "types_worse_by_more_than_1min": worse,
               "overall_share_within_10_drop_points": round(100 * share_drop, 2),
               "adopt_lookback_2": bool(fixed and not worse and share_drop <= 0.01)}
    print("\nVERDICT", verdict)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/processed/lookback_test.json").write_text(json.dumps(
        {"lookback1": json.loads(o1.to_json()), "lookback2": json.loads(o2.to_json()), "overall1": a1, "overall2": a2,
         "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
