"""Primary delay sources: incidents, halt extensions, entry delays, freight blockers, disruptions.

SIMULATED. All parameters come from config.yaml. Per-run draws use the run's own random stream so a
train's draws do not depend on how other trains interleave.
"""
import itertools
import math

import numpy as np

from . import causes as C
from .weather import day_to_date

EVENT_TYPES = {1: "section_incident", 2: "halt_extension", 3: "entry_delay",
               4: "disruption_block", 5: "disruption_signal", 6: "freight"}


class DelayModel:
    """Implements the delay interface used by the engine (see engine.NullDelays)."""

    def __init__(self, net, cfg, weather, seed):
        self.net, self.cfg, self.weather, self.seed = net, cfg, weather, seed
        self.inc, self.hx = cfg["incidents"], cfg["halt_extension"]
        self.wx, self.en = cfg["weather"], cfg["entry_delay"]
        self.fr, self.dis = cfg["freight"], cfg["disruptions"]
        self.ids = itertools.count(1)
        self.events = []                     # (id, day, type, section, node, start, duration, train, direction)
        self.blocks = [[] for _ in range(net.n_sections)]   # per section: (start, end, id, code, dir_mask)
        self._peak = set(self.inc["peak_hours"])
        self._month_cache = {}

    # ------------------------------------------------------------------ helpers
    def _month(self, day):
        m = self._month_cache.get(day)
        if m is None:
            m = self._month_cache[day] = day_to_date(day).month
        return m

    def _hour_factor(self, t):
        return self.inc["peak_hour_factor"] if int((t % 1440.0) // 60) in self._peak else 1.0

    def _log(self, day, typ, sec, node, start, dur, train, direction=0):
        eid = next(self.ids)
        self.events.append((eid, day, typ, sec, node, start, dur, train, direction))
        return eid

    # ------------------------------------------------------------------ engine interface
    def weather_factor(self, sec, t):
        s = self.weather.state(int(t // 1440))
        if s == "fog":
            return 1.0 + self.wx["fog_slowdown"] * self.net.section_fog[sec]
        if s == "rain":
            return 1.0 + self.wx["rain_slowdown"]
        return 1.0

    def _incident_minutes(self, rnd, p_tail, median, sigma, xm, alpha, cap):
        if rnd.random() < p_tail:
            x = xm * rnd.paretovariate(alpha)
        else:
            x = rnd.lognormvariate(math.log(median), sigma)
        return min(x, cap)

    def incident(self, run, sec, t):
        day = int(t // 1440)
        p = (self.inc["base_rate"] * self.net.section_node_factor[sec] * self._hour_factor(t)
             * self.inc["month_factor"].get(self._month(day), 1.0)
             * self.inc["weather_factor"][self.weather.state(day)])
        if run.rnd.random() >= p:
            return 0.0, 0
        i = self.inc
        x = self._incident_minutes(run.rnd, i["tail_probability"], i["lognormal_median_min"], i["lognormal_sigma"],
                                   i["pareto_min_min"], i["pareto_alpha"], i["cap_min"])
        return x, self._log(day, 1, sec, -1, t, x, run.spec.train)

    def halt_extension(self, run, node, t):
        day = int(t // 1440)
        p = (self.hx["probability"] * self._hour_factor(t) * self.inc["month_factor"].get(self._month(day), 1.0))
        if run.rnd.random() >= p:
            return 0.0, 0
        x = min(run.rnd.lognormvariate(math.log(self.hx["lognormal_median_min"]), self.hx["lognormal_sigma"]),
                self.hx["cap_min"])
        return x, self._log(day, 2, -1, node, t, x, run.spec.train)

    def entry_delay(self, run, t):
        e = self.en
        if run.rnd.random() < e["zero_probability"]:
            return 0.0, 0
        day = int(t // 1440)
        x = run.rnd.lognormvariate(math.log(e["lognormal_median_min"]), e["lognormal_sigma"])
        x *= 1.0 + e["journey_fraction_weight"] * run.spec.extra.get("journey_fraction", 0.0)
        x *= e["type_factor"].get(run.spec.type, 1.0)
        x *= self.inc["month_factor"].get(self._month(day), 1.0)
        if self.weather.state(day) == "fog":
            x *= e["fog_multiplier"]
        x = min(x, e["cap_min"])
        return x, self._log(day, 3, -1, -1, t, x, run.spec.train)

    def block_end(self, sec, direction, t, priority):
        """(end, cause code, event id) of the latest-ending active block that applies, or None."""
        best = None
        for start, end, eid, code, mask in self.blocks[sec]:
            if start <= t < end and (mask == 0 or mask == direction):
                if code == C.FREIGHT and priority <= self.fr["yield_priority"]:
                    continue
                if best is None or end > best[0]:
                    best = (end, code, eid)
        return best

    # ------------------------------------------------------------------ day-level generation
    def generate_day(self, day):
        """Freight blockers and disruptions that start during calendar day `day`."""
        rng = np.random.default_rng([self.seed, 7, day + 1000])
        t0 = day * 1440.0
        for lst in self.blocks:                       # forget blocks that ended before this day
            lst[:] = [b for b in lst if b[1] > t0 - 1.0]
        n = self.net.n_sections
        rate = np.full(n, self.fr["rate_per_section_hour"])
        rate[np.array(self.net.section_congested, bool)] *= self.fr["junction_multiplier"]
        counts = rng.poisson(rate * 24.0)
        for sec in np.nonzero(counts)[0]:
            for _ in range(counts[sec]):
                start = t0 + rng.uniform(0, 1440.0)
                dur = float(min(rng.lognormal(math.log(self.fr["duration_median_min"]), self.fr["duration_sigma"]), 60.0))
                eid = self._log(day, 6, int(sec), -1, start, dur, "")
                self.blocks[sec].append((start, start + dur, eid, C.FREIGHT, 0))
        dcounts = rng.poisson(self.dis["rate_per_section_day"], n)
        for sec in np.nonzero(dcounts)[0]:
            for _ in range(dcounts[sec]):
                start = t0 + rng.uniform(0, 1440.0)
                dur = float(np.clip(rng.lognormal(math.log(self.dis["duration_median_min"]), self.dis["duration_sigma"]),
                                    self.dis["duration_min"], self.dis["duration_max"]))
                signal = rng.random() < self.dis["signal_failure_share"]
                mask = int(rng.choice([-1, 1])) if signal else 0
                eid = self._log(day, 5 if signal else 4, int(sec), -1, start, dur, "", mask)
                self.blocks[sec].append((start, start + dur, eid, C.DISRUPTION, mask))
