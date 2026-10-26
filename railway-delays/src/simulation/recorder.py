"""Collects observations / attribution rows during a run and writes the output tables (SIMULATED)."""
import json
import math
from array import array
from pathlib import Path

import numpy as np
import pandas as pd

from . import causes as C
from .delays import EVENT_TYPES
from .weather import day_to_date

ON_TIME_EPS = 0.5          # minutes: below this a row counts as on time
NEW_CAUSES = (C.INCIDENT, C.DISRUPTION, C.WEATHER, C.SECTION_WAIT, C.YIELD, C.FREIGHT, C.RAKE, C.HALT_EXT, C.ENTRY)
INJECTED = (C.INCIDENT, C.HALT_EXT, C.ENTRY)

OBS_COLS = [("obs_id", "q"), ("day", "i"), ("train", "i"), ("station", "i"), ("seq", "h"),
            ("scheduled_arrival", "f"), ("scheduled_departure", "f"), ("raw_scheduled_arrival", "f"),
            ("raw_scheduled_departure", "f"), ("actual_arrival", "f"), ("actual_departure", "f"),
            ("arrival_delay", "f"), ("departure_delay", "f"), ("primary_delay_injected", "f"),
            ("cause_code", "b"), ("cause_train", "i"), ("root_event_id", "i")]
ATTR_COLS = [("obs_id", "q"), ("phase", "b"), ("cause_code", "b"), ("minutes", "f"), ("new_minutes", "f"),
             ("cause_train", "i"), ("event_id", "i"), ("root_event_id", "i")]
NAN = float("nan")


class Buffer:
    def __init__(self, cols):
        self.cols = cols
        self.data = {n: array(t) for n, t in cols}

    def append(self, *vals):
        for (n, _), v in zip(self.cols, vals):
            self.data[n].append(v)

    def __len__(self):
        return len(self.data[self.cols[0][0]])

    def numpy(self, name):
        a = self.data[name]
        return np.frombuffer(a, dtype={"q": np.int64, "i": np.int32, "h": np.int16, "f": np.float32, "b": np.int8}[a.typecode]) \
            if len(a) else np.empty(0)


class Recorder:
    def __init__(self, specs, net, start_day, end_day):
        self.net, self.start_day, self.end_day = net, start_day, end_day
        self.trains = sorted(specs)
        self.train_code = {t: i for i, t in enumerate(self.trains)}
        stations = sorted({r[0] for s in specs.values() for r in s.extra["rows"]})
        self.stations = stations
        self.station_code = {s: i for i, s in enumerate(stations)}
        self.obs, self.attr = {}, {}
        self.pending = {}           # rid -> {"arr": {i: rec}, "dep": {i: rec}}
        self.next_obs = 0
        self.run_count = 0

    # called by the ledger at every arrival / departure commit
    def visit_obs(self, run, i, phase, t, comps, added, rec_since):
        d = self.pending.setdefault(run.rid, ({}, {}))
        d[0 if phase == "arr" else 1][i] = (t, comps, added, rec_since)

    def _month(self, day):
        dt = day_to_date(day)
        return f"{dt.year}-{dt.month:02d}"

    def finish(self, run):
        """Turn the stored visit records of a finished run into observation and attribution rows."""
        rec = self.pending.pop(run.rid, None)
        if rec is None or not (self.start_day <= run.day <= self.end_day):
            return
        self.run_count += 1
        month = self._month(run.day)
        ob = self.obs.setdefault(month, Buffer(OBS_COLS))
        at = self.attr.setdefault(month, Buffer(ATTR_COLS))
        spec = run.spec
        arr_rec, dep_rec = rec
        tcode = self.train_code[spec.train]
        t0 = run.t0
        for (station, seq, k, j, n, s_arr, s_dep, r_arr, r_dep) in spec.extra["rows"]:
            v = spec.visits[k]
            a_t, a_comps, a_added, a_rec = arr_rec[k]
            arr_delay = a_t - (t0 + v.sched_arr)
            if k in dep_rec:
                d_t = dep_rec[k][0]
                dep_delay = d_t - (t0 + v.sched_dep)
            else:
                dep_delay = arr_delay
            row_dep_delay = dep_delay if j == n - 1 else arr_delay
            has_arr = not math.isnan(r_arr)
            has_dep = not math.isnan(r_dep)
            cause, who, root, injected = C.NONE, -1, 0, 0.0
            if j == 0:
                cause, who, root, injected = self._dominant(a_added, arr_delay)
            elif arr_delay >= ON_TIME_EPS:
                cause = C.INHERITED
            oid = self.next_obs
            self.next_obs += 1
            day = run.day
            ob.append(oid, day, tcode, self.station_code[station], seq, s_arr, s_dep, r_arr, r_dep,
                      s_arr + arr_delay if has_arr else NAN, s_dep + row_dep_delay if has_dep else NAN,
                      arr_delay if has_arr else NAN, row_dep_delay if has_dep else NAN, injected, cause, who, root)
            if j == 0 and has_arr and arr_delay >= ON_TIME_EPS:
                self._attr_rows(at, oid, 0, a_comps, a_added, a_rec)
            if j == n - 1 and k in dep_rec and has_dep and (v.halt or abs(dep_delay - arr_delay) > 0.01):
                _, d_comps, _, d_rec = dep_rec[k]
                if dep_delay >= ON_TIME_EPS:
                    self._attr_rows(at, oid, 1, d_comps, None, 0.0)

    def _dominant(self, added, delay):
        """(cause code, causing train code, root event, injected minutes) for the minutes added since the previous row."""
        injected, best, best_m = 0.0, None, 0.01
        for key, m in (added or {}).items():
            if key[0] in INJECTED:
                injected += m
            if m > best_m and key[0] in NEW_CAUSES:
                best, best_m = key, m
        if best is not None:
            return best[0], self.train_code.get(best[1], -1), best[3], injected
        if delay >= ON_TIME_EPS:
            return C.INHERITED, -1, 0, injected
        return C.NONE, -1, 0, injected

    def _attr_rows(self, at, oid, phase, comps, added, rec_since):
        for (code, who, eid, root), m in comps:
            new = (added or {}).get((code, who, eid, root), 0.0)
            at.append(oid, phase, code, m, new, self.train_code.get(who, -1), eid, root)
        if phase == 0 and rec_since > 0.01:
            at.append(oid, phase, C.RECOVERY, -rec_since, 0.0, -1, 0, 0)

    # ------------------------------------------------------------------ output
    def tables(self, month):
        """DataFrames (observations, attribution) for one month."""
        ob, at = self.obs.get(month), self.attr.get(month)
        o = {n: ob.numpy(n) for n, _ in OBS_COLS} if ob else {n: np.empty(0) for n, _ in OBS_COLS}
        base = pd.Timestamp("2025-01-01")
        df = pd.DataFrame({
            "obs_id": o["obs_id"],
            "date": base + pd.to_timedelta(o["day"], unit="D"),
            "train": pd.Categorical.from_codes(o["train"].astype(np.int64), self.trains) if len(o["train"]) else pd.Categorical([]),
            "station": pd.Categorical.from_codes(o["station"].astype(np.int64), self.stations) if len(o["station"]) else pd.Categorical([]),
            "seq": o["seq"],
            **{n: o[n] for n in ("scheduled_arrival", "scheduled_departure", "raw_scheduled_arrival",
                                 "raw_scheduled_departure", "actual_arrival", "actual_departure", "arrival_delay",
                                 "departure_delay", "primary_delay_injected", "cause_code")},
            "cause_train": pd.Categorical.from_codes(o["cause_train"].astype(np.int64), self.trains) if len(o["cause_train"]) else pd.Categorical([]),
            "root_event_id": o["root_event_id"],
            "is_simulated": True,
        })
        a = {n: at.numpy(n) for n, _ in ATTR_COLS} if at else {n: np.empty(0) for n, _ in ATTR_COLS}
        dfa = pd.DataFrame({
            "obs_id": a["obs_id"],
            "phase": pd.Categorical.from_codes(a["phase"].astype(np.int64), ["arr", "dep"]) if len(a["phase"]) else pd.Categorical([]),
            "cause_code": a["cause_code"], "minutes": a["minutes"], "new_minutes": a["new_minutes"],
            "cause_train": pd.Categorical.from_codes(a["cause_train"].astype(np.int64), self.trains) if len(a["cause_train"]) else pd.Categorical([]),
            "event_id": a["event_id"], "root_event_id": a["root_event_id"], "is_simulated": True,
        })
        return df, dfa

    def months(self):
        return sorted(self.obs)


def write_outputs(recorder, delay_model, weather, cfg, out_dir, start_day, end_day, extra_meta=None):
    """Write observations / delay_attribution (partitioned by month), events, weather and config_used.yaml."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    out = Path(out_dir)
    meta = {"SIMULATED": "true", "note": "Synthetic data from the Corridor B delay simulator; not real delays."}
    meta.update(extra_meta or {})
    sizes = {}

    def write(df, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        t = pa.Table.from_pandas(df, preserve_index=False)
        t = t.replace_schema_metadata({**(t.schema.metadata or {}), **{k.encode(): str(v).encode() for k, v in meta.items()}})
        pq.write_table(t, path, compression="zstd")
        sizes[str(path.relative_to(out))] = path.stat().st_size

    for month in recorder.months():
        df, dfa = recorder.tables(month)
        write(df, out / "observations" / f"month={month}" / "part.parquet")
        write(dfa, out / "delay_attribution" / f"month={month}" / "part.parquet")
    rows = []
    for e in delay_model.events:
        eid, day, typ, sec, node, start, dur, train, direction = e
        if not (start_day <= day <= end_day):
            continue
        name = ""
        if sec >= 0:
            name = f"{recorder.net.node_names[sec]}-{recorder.net.node_names[sec + 1]}"
        elif node >= 0:
            name = recorder.net.node_names[node]
        rows.append((eid, day_to_date(day), EVENT_TYPES[typ], sec, node, name, float(start), float(dur), train, direction))
    ev = pd.DataFrame(rows, columns=["event_id", "date", "type", "section", "node", "place", "start_min", "duration_min",
                                     "train", "direction"])
    ev["date"] = pd.to_datetime(ev["date"])
    ev["type"] = ev["type"].astype("category")
    ev["duration_min"] = ev["duration_min"].astype("float32")
    ev["is_simulated"] = True
    write(ev, out / "events.parquet")
    wd = pd.DataFrame({"date": [pd.Timestamp(day_to_date(d)) for d in range(start_day, end_day + 1)],
                       "state": [weather.state(d) for d in range(start_day, end_day + 1)],
                       "source": weather.source, "is_simulated": True})
    wd["state"] = wd["state"].astype("category")
    write(wd, out / "weather.parquet")
    (out / "config_used.yaml").write_text("# SIMULATED data: configuration used for this run\n" + yaml.safe_dump(cfg, sort_keys=False))
    (out / "README_SIMULATED.txt").write_text(
        "SIMULATED DATA. Produced by src/simulation from the Corridor B timetable; nothing here is real delay data.\n"
        "Times in observations are minutes from 00:00 of `date` (the date the train enters the corridor).\n")
    return sizes
