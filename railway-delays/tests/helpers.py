"""Tiny hand-built networks and trains for the engine tests."""
import random

from src.simulation.engine import Hop, MoveParams, Net, NullDelays, Simulator, TrainSpec, Visit


def make_net(n_nodes=4, junctions=(), headway=5.0):
    return Net([f"N{i}" for i in range(n_nodes)], [i in junctions for i in range(n_nodes)],
               [10.0] * (n_nodes - 1), [headway] * (n_nodes - 1))


def line_spec(train, prio, t_start, section_minutes, first=0, last=3, halts=None, slack=0.06,
              min_halt_factor=0.5, entered=False, feeder=None, typ="Superfast"):
    """A train running first -> last. `section_minutes` scheduled per section (number or list).
    `halts` {node: dwell minutes}; other nodes are pass-through."""
    halts = halts or {}
    d = 1 if last > first else -1
    nodes = list(range(first, last + d, d))
    sm = section_minutes if isinstance(section_minutes, list) else [section_minutes] * (len(nodes) - 1)
    visits, hops, t = [], [], t_start
    for k, n in enumerate(nodes):
        dwell = halts.get(n, 0.0)
        mh = min(dwell, max(1.0, round(min_halt_factor * dwell))) if dwell > 0 else 0.0
        visits.append(Visit(n, t, t + dwell, dwell > 0 or k == 0 or k == len(nodes) - 1, mh))
        t += dwell
        if k < len(nodes) - 1:
            sec = n if d == 1 else n - 1
            hops.append(Hop("section", (sec,), (sm[k],), d))
            t += sm[k]
    return TrainSpec(train, typ, prio, slack, visits, hops, entered_mid_route=entered, feeder=feeder)


def sim(net, mp=None, delays=None):
    return Simulator(net, mp or MoveParams(), delays or NullDelays())


def run(sim_, spec, day=0, seed=1):
    return sim_.add_run(spec, day, random.Random(seed))


def delays_at(run_):
    """(arrival delay, departure delay) per visit."""
    out = []
    for v, a, d in zip(run_.spec.visits, run_.arr, run_.dep):
        out.append((round(a - (run_.t0 + v.sched_arr), 6), round(d - (run_.t0 + v.sched_dep), 6)))
    return out


class FixedEntry(NullDelays):
    """Entry delay of a fixed number of minutes."""

    def __init__(self, minutes):
        self.minutes = minutes

    def entry_delay(self, run, t):
        return self.minutes, 1
