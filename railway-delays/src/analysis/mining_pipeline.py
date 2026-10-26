"""The frozen mining pipeline (SIMULATED data, observable inputs only).

    run_pipeline(start, end, scenario) loads the window, builds passages, applies the FROZEN parameters
    (configs/mining_frozen.yaml, calibrated on 2024-01-01..2024-10-31 only) and returns / saves every table. Nothing is
    re-estimated from the window it is applied to, except the descriptive statistics the tables report.
It never reads delay_attribution or the ground-truth cascades.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import cascades as cs
from src.analysis import contagion as ct
from src.analysis import decomposition as dc
from src.analysis import criticality as cr
from src.analysis import eda, passages
from src.analysis.calibrate import load_frozen
from src.graph import train_graph as tgm
from src.graph.station_graph import build_station_tables
from src.graph.timetable import load_timetable

OUT = Path("data/processed/mining")


def window_tag(start, end):
    return f"{start}_{end}"


def run_pipeline(start, end, scenario="medium", params=None, save=True, verbose=True, tt=None, nodes=None):
    """Apply the frozen pipeline to [start, end]. Returns a dict of tables."""
    prm = params or load_frozen()
    t0 = time.time()
    tt = tt or load_timetable()
    nodes, edges, _ = build_station_tables(tt)
    df, tt, nodes, bounds = eda.load_window(start, end, scenario=scenario, tt=tt, nodes=nodes)
    p = passages.attach_leaders(passages.build_passages(df, bounds))
    runs = passages.build_runs(df, tt, bounds)
    hourly = eda.node_hour_arrays(eda.arrivals(df, bounds), len(nodes), int((bounds[1] - bounds[0]) // 60))
    del df
    p = dc.decompose(p, prm.slack_fraction)
    explained = dc.flag_explained(p, prm)
    primary = (p.added >= prm.P) & ~explained
    if verbose:
        print(f"[{scenario} {start}..{end}] {len(p):,} passages, {len(runs):,} runs prepared in {time.time() - t0:.0f}s", flush=True)

    out = {"params": prm, "n_passages": len(p), "n_runs": len(runs), "bounds": bounds, "window": (start, end), "scenario": scenario}
    # item 1
    p["explained"], p["primary"] = explained.to_numpy(), primary.to_numpy()
    out["added_by_section"] = dc.added_by(p, ["sec", "dir"]).reset_index()
    out["added_by_type"] = dc.added_by(p, ["type"]).reset_index()
    out["added_by_type_section"] = dc.added_by(p, ["type", "sec"]).reset_index()
    out["primary_events"] = p[p.primary][["run_id", "train", "type", "sec", "dir", "t_in", "added", "weather", "month", "d_in"]].copy()
    out["decomposition_totals"] = pd.Series({"inherited_min": float(p.inherited.sum()), "added_min": float(p.added.clip(lower=0).sum()),
                                             "absorbed_min": float(p.absorbed.sum()), "explained_share_of_added": float(p.added.clip(lower=0)[p.explained].sum() / p.added.clip(lower=0).sum()),
                                             "primary_events": int(p.primary.sum()), "passages": len(p)})
    # item 2
    cc = ct.contagion_coefficients(p, prm.G)
    po = ct.pass_on_ratio(p, explained, primary, prm)
    out["contagion"] = cc.merge(po.drop(columns=["sec", "dir"]), on="key", how="outer")
    out["slack_absorbed"] = ct.slack_absorbed(p)
    out["slack_absorbed_type"] = ct.slack_absorbed(p, ("type",))
    # item 3
    casc, members, links = cs.extract_cascades(p, runs, prm, explained, primary)
    out["cascades"], out["members"], out["links"] = casc, members, links
    out["passage_links"] = p[explained & (p.added >= prm.A_link) & p.lead_run.notna()][
        ["run_id", "train", "type", "to_node", "from_node", "sec", "dir", "t_in", "t_out", "added", "lead_run", "lead_train", "gap_act"]].copy()
    out["runs"] = runs.reset_index()
    out["hourly_mean_delay"], out["hourly_count"] = hourly
    out["nodes"], out["edges"] = nodes, edges
    out.update(item4(out, tt))
    if verbose:
        print(f"[{scenario} {start}..{end}] done in {time.time() - t0:.0f}s: {len(casc):,} cascades", flush=True)
    del p
    if save:
        d = OUT / scenario / window_tag(start, end)
        d.mkdir(parents=True, exist_ok=True)
        for k, v in out.items():
            if isinstance(v, pd.DataFrame):
                v.to_parquet(d / f"{k}.parquet", index=False)
        np.savez_compressed(d / "hourly.npz", mean=out["hourly_mean_delay"], count=out["hourly_count"])
    return out


SHATABDI_ARTIFACT = ("12003", "12033")


def item4(out, tt):
    """Critical sections and trains (needs the cascades, links, contagion table and the two timetable graphs)."""
    tn, te = tgm.build_train_tables(tt)
    sg = cr.station_digraph(out["nodes"], out["edges"], out["contagion"])
    tgr = cr.train_digraph(te, tn)
    through = {}
    for train, sp in tt.specs.items():
        for h in sp.hops:
            if h.kind == "section":
                for s in h.sections:
                    through.setdefault(s, []).append(train)
    secs = cr.critical_sections(out["cascades"], out["primary_events"], out["nodes"], sg, tgr, through, tt.net.n_sections)
    names = tt.net.node_names
    secs["section"] = [f"{names[s]}-{names[s + 1]}" for s in secs.sec]
    trains = cr.critical_trains(out["cascades"], out["links"], out["members"], tgr)
    trains["name"] = trains.train.map(tt.trains.set_index("number").name)
    trains["type"] = trains.train.map({t: s.type for t, s in tt.specs.items()})
    kns = min(names.index("KNS"), names.index("PJY"))
    flags = cr.flag_artifacts(secs, out["primary_events"], {"KNS-PJY": kns}, set(SHATABDI_ARTIFACT))
    return {"critical_sections": secs, "critical_trains": trains, "artifact_flags": flags}


def load_saved(start, end, scenario="medium"):
    d = OUT / scenario / window_tag(start, end)
    return {f.stem: pd.read_parquet(f) for f in d.glob("*.parquet")}
