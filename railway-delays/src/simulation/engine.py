"""Event-driven train movement engine (SIMULATED data).

Times are minutes. Each run has a local frame (minutes from 00:00 of its entry date); its absolute
time is run.t0 + local. One lane per direction; sections are consecutive nodes on the chain.

Rules (see reports/phase2b_simulator_design.md):
* departure = max(scheduled departure, arrival + min halt, next section free, rake ready)
* headway between entries into a section, same direction; no passing inside a section
* priority yield and overtaking only at junctions or where the slower train has a real halt
* slack recovery up to slack x scheduled section time; halts shorten to min halt
* off-corridor legs are fixed-time, carry delay, and never conflict
"""
import heapq
import itertools
from dataclasses import dataclass, field

from . import causes as C

EPS = 1e-9


# --------------------------------------------------------------------------- data classes
@dataclass
class Visit:
    node: int
    sched_arr: float
    sched_dep: float
    halt: bool = False      # real halt: may be shortened to min_halt (otherwise dwell is fixed)
    min_halt: float = 0.0


@dataclass
class Hop:
    kind: str                       # "section" or "leg"
    sections: tuple = ()            # section indices in travel order
    shares: tuple = ()              # scheduled minutes per section
    direction: int = 1              # +1 / -1 along the chain
    leg_minutes: float = 0.0


@dataclass
class TrainSpec:
    train: str
    type: str
    priority: int
    slack: float
    visits: list
    hops: list
    entered_mid_route: bool = False
    exits_mid_route: bool = False
    feeder: tuple = None            # (feeder train, entry-day difference k, scheduled turnaround gap)
    extra: dict = field(default_factory=dict)   # rows, frame shift, etc. (used by the recorder)


@dataclass
class Net:
    node_names: list
    node_junction: list
    section_km: list
    headway: list                   # minutes per section
    section_node_factor: list = None
    section_congested: list = None
    section_fog: list = None        # fog intensity 0..1 per section

    @property
    def n_sections(self):
        return len(self.section_km)


@dataclass
class MoveParams:
    exit_gap: float = 2.0
    yield_window: float = 10.0
    max_yield_hold: float = 30.0
    lookback: int = 1
    yield_min_conflict: float = 1.0
    overtake_mode: str = "junction_or_halt"   # or "any_station" (config default)
    loop_capacity: int = 2                   # any_station mode: trains held aside per station
    overtake_penalty: float = 2.0            # any_station mode: extra minutes paid by the overtaken train
    min_turnaround: float = 45.0
    offcorridor_recovery: float = 0.3
    rake_noise_sd: float = 3.0


class NullLedger:
    """Does nothing; the real ledger (cause tracking) lives in ledger.py."""

    def add(self, run, code, minutes, cause_train="", event_id=0, root=0):
        pass

    def recover(self, run, minutes):
        pass

    def root_of(self, run):
        return 0

    def obs(self, run, i, phase):
        pass


class NullDelays:
    """No weather, incidents, freight or disruptions."""

    def weather_factor(self, sec, t):
        return 1.0

    def incident(self, run, sec, t):
        return 0.0, 0

    def halt_extension(self, run, node, t):
        return 0.0, 0

    def entry_delay(self, run, t):
        return 0.0, 0

    def block_end(self, sec, direction, t, priority):
        return None


def rake_delay(feeder_exit_delay, gap_sched, min_turnaround, feeder_exits_mid, train_entered_mid,
               recovery, noise=0.0):
    """Delay passed on to the next service of a rake (design section 3.6).

    The feeding train's delay at the corridor edge is reduced by off-corridor recovery if it still has
    to run off-corridor, then by the scheduled turnaround buffer. The result is reduced again if the
    next train runs off-corridor before reaching the corridor.
    """
    d = max(feeder_exit_delay, 0.0)
    if feeder_exits_mid:
        d = max(0.0, d * (1 - recovery) + noise)
    buffer = max(0.0, gap_sched - min_turnaround)
    d = max(0.0, d - buffer)
    if train_entered_mid:
        d *= 1 - recovery
    return d


class Run:
    __slots__ = ("spec", "day", "t0", "rid", "key", "idx", "arr", "dep", "ready", "yield_held",
                 "inbound_key", "rnd", "done", "exit_delay", "started", "entry_delay", "meta", "wait_acc", "held_at")

    def __init__(self, spec, day, rid, rnd):
        self.spec, self.day, self.rid, self.rnd = spec, day, rid, rnd
        self.t0 = day * 1440.0
        self.key = (spec.train, day)
        n = len(spec.visits)
        self.idx = 0
        self.arr = [None] * n
        self.dep = [None] * n
        self.ready = None
        self.yield_held = 0.0
        self.inbound_key = None
        self.done = False
        self.exit_delay = 0.0
        self.started = False
        self.entry_delay = 0.0
        self.meta = {}
        self.wait_acc = 0.0
        self.held_at = None


class Simulator:
    def __init__(self, net, mp=None, delays=None, ledger=None, on_finish=None):
        self.net = net
        self.mp = mp or MoveParams()
        self.delays = delays or NullDelays()
        self.ledger = ledger or NullLedger()
        self.on_finish = on_finish
        self.heap = []
        self._cnt = itertools.count()
        self._rid = itertools.count(1)
        self.last_entry = {}     # (sec, dir) -> (time, run)
        self.last_arr = {}       # (sec, dir) -> (time, run)
        self.inbound = {}        # (node, dir) -> {rid: run}
        self.runs = {}           # (train, day) -> run
        self.now = 0.0
        self.held = {}           # node -> trains currently held aside for an overtake (any_station mode)
        self.queue_log = None   # set to a list to record (section, direction, wait start, wait end, train)
        self.pairs = {}          # scheduled pair spacing, see network.scheduled_pair_gaps

    # ------------------------------------------------------------------ public API
    def add_run(self, spec, day, rnd=None):
        run = Run(spec, day, next(self._rid), rnd)
        self.runs[run.key] = run
        self._push(run.t0 + spec.visits[0].sched_arr, 0, run)
        return run

    def run_until(self, t_end):
        heap = self.heap
        while heap and heap[0][0] < t_end:
            t, _, kind, run = heapq.heappop(heap)
            self.now = t
            if kind == 0:
                self._start(run, t)
            else:
                self._depart(run, t)

    def drain(self):
        self.run_until(float("inf"))

    def forget_before(self, day):
        for k in [k for k in self.runs if k[1] < day and self.runs[k].done]:
            del self.runs[k]

    # ------------------------------------------------------------------ internals
    def _push(self, t, kind, run):
        heapq.heappush(self.heap, (t, next(self._cnt), kind, run))

    def _start(self, run, T):
        spec = run.spec
        v0 = spec.visits[0]
        sched_arr = run.t0 + v0.sched_arr
        delay = 0.0
        if spec.feeder is not None:
            ftrain, k, gap = spec.feeder
            frun = self.runs.get((ftrain, run.day - k))
            if frun is not None:
                if not frun.done:                       # wait for the feeding train to reach the edge
                    self._push(max(T + 1.0, frun.ready or 0.0), 0, run)
                    return
                noise = run.rnd.gauss(0.0, self.mp.rake_noise_sd) if (run.rnd and frun.spec.exits_mid_route) else 0.0
                rk = rake_delay(frun.exit_delay, gap, self.mp.min_turnaround, frun.spec.exits_mid_route,
                                spec.entered_mid_route, self.mp.offcorridor_recovery, noise)
                if rk > EPS:
                    self.ledger.add(run, C.RAKE, rk, ftrain, 0, self.ledger.root_of(frun))
                    delay += rk
        if spec.entered_mid_route:
            x, eid = self.delays.entry_delay(run, T)
            if x > EPS:
                self.ledger.add(run, C.ENTRY, x, "", eid, eid)
                delay += x
        run.entry_delay = delay
        t_arr = sched_arr + delay
        if t_arr < T:                                   # held back by an unfinished feeder
            extra = T - t_arr
            self.ledger.add(run, C.RAKE, extra, spec.feeder[0] if spec.feeder else "", 0, 0)
            t_arr = T
        run.started = True
        self._arrive(run, 0, t_arr)

    def _arrive(self, run, i, t_arr):
        spec = run.spec
        v = spec.visits[i]
        run.idx = i
        run.arr[i] = t_arr
        self.ledger.obs(run, i, "arr")
        if i == len(spec.visits) - 1:
            run.dep[i] = t_arr
            run.exit_delay = t_arr - (run.t0 + v.sched_arr)
            run.done = True
            if self.on_finish:
                self.on_finish(run)
            return
        d_arr = t_arr - (run.t0 + v.sched_arr)
        sched_dep = run.t0 + v.sched_dep
        dwell = v.sched_dep - v.sched_arr
        if v.halt:
            ready = max(sched_dep, t_arr + v.min_halt)
        else:
            ready = t_arr + dwell
        delta = (ready - sched_dep) - d_arr
        if delta < -EPS:
            self.ledger.recover(run, -delta)
        if v.halt and dwell > EPS:
            ext, eid = self.delays.halt_extension(run, v.node, t_arr)
            if ext > EPS:
                self.ledger.add(run, C.HALT_EXT, ext, "", eid, eid)
                ready += ext
        run.ready = ready
        hop = spec.hops[i]
        if run.inbound_key is not None:
            self.inbound.get(run.inbound_key, {}).pop(run.rid, None)
            run.inbound_key = None
        if hop.kind == "section":
            run.inbound_key = (v.node, hop.direction)
            self.inbound.setdefault(run.inbound_key, {})[run.rid] = run
        self._push(ready, 1, run)

    def _commit_departure(self, run, T, sec=None, direction=0):
        i = run.idx
        if run.wait_acc > EPS:
            if self.queue_log is not None and sec is not None:
                self.queue_log.append((sec, direction, T - run.wait_acc, T, run.spec.train))
            run.wait_acc = 0.0
        run.dep[i] = T
        if run.inbound_key is not None:
            self.inbound.get(run.inbound_key, {}).pop(run.rid, None)
            run.inbound_key = None
        self.ledger.obs(run, i, "dep")

    def _depart(self, run, T):
        spec = run.spec
        i = run.idx
        v = spec.visits[i]
        hop = spec.hops[i]
        run.ready = T
        if hop.kind == "leg":
            self._commit_departure(run, T)
            d = max(0.0, T - (run.t0 + v.sched_dep))
            self._arrive(run, i + 1, run.t0 + spec.visits[i + 1].sched_arr + d)
            return
        sec, direction = hop.sections[0], hop.direction
        # 1. yield (be overtaken) by a higher-priority train that is about to enter the same section
        node = v.node
        if run.held_at is not None:                       # the previous hold at a loop has ended
            self.held[run.held_at] -= 1
            run.held_at = None
        any_mode = self.mp.overtake_mode == "any_station"
        if any_mode:
            can_yield = self.held.get(node, 0) < self.mp.loop_capacity
        else:
            can_yield = self.net.node_junction[node] or v.halt
        if run.yield_held < self.mp.max_yield_hold - EPS and can_yield:
            found = self._priority_inbound(run, i, T)
            if found is not None:
                h_run, h_ready = found
                req = self.net.headway[sec]
                pg = self.pairs.get((sec, direction))
                if pg:
                    req = pg.get((spec.train, h_run.spec.train), (req, 0))[0]
                pen = self.mp.overtake_penalty if any_mode else 0.0
                hold = min(h_ready + req - T + pen, self.mp.max_yield_hold - run.yield_held)
                if hold > EPS:
                    self.ledger.add(run, C.YIELD, hold, h_run.spec.train, 0, self.ledger.root_of(h_run))
                    run.yield_held += hold
                    run.wait_acc += hold
                    if any_mode:
                        run.held_at = node
                        self.held[node] = self.held.get(node, 0) + 1
                    self._push(T + hold, 1, run)
                    return
        # 2. headway / freight / disruption
        t_e, cause = self._entry_time(run, sec, direction, T)
        if t_e > T + EPS:
            code, who, eid, root = cause
            self.ledger.add(run, code, t_e - T, who, eid, root)
            run.wait_acc += t_e - T
            self._push(t_e, 1, run)
            return
        self._commit_departure(run, T, sec, direction)
        self._traverse(run, i, T)

    def _entry_time(self, run, sec, direction, T):
        """Earliest time `run` may enter `sec`, and the binding cause (code, train, event id, root)."""
        last = self.last_entry.get((sec, direction))
        if last:
            req = self.net.headway[sec]
            pg = self.pairs.get((sec, direction))
            if pg:
                req = pg.get((run.spec.train, last[1].spec.train), (req, 0))[0]
            t_h = last[0] + req
        else:
            t_h = float("-inf")
        blk = self.delays.block_end(sec, direction, T, run.spec.priority)
        t_b = blk[0] if blk else float("-inf")
        t_e = max(T, t_h, t_b)
        if t_e <= T + EPS:
            return T, None
        if t_b >= t_h:
            return t_e, (blk[1], "", blk[2], blk[2])
        lrun = last[1]
        return t_e, (C.SECTION_WAIT, lrun.spec.train, 0, self.ledger.root_of(lrun))

    def _priority_inbound(self, run, i, T):
        """Higher-priority train that letting `run` go first would delay by more than yield_min_conflict.

        Candidates are trains that will be ready to enter the same section within the window, seen up to
        `lookback` sections behind (for depth > 1 their arrival is an estimate at 95% of scheduled time).
        Returns (train, ready time at this station) for the earliest such train, or None.
        """
        spec = run.spec
        hop = spec.hops[i]
        sec, direction = hop.sections[0], hop.direction
        node = spec.visits[i].node
        sched_dep = run.t0 + spec.visits[i].sched_dep
        l_share = hop.shares[0]
        l_arr = T + l_share - min(max(0.0, T - sched_dep), spec.slack * l_share)
        best = None
        for m in range(self.mp.lookback):
            reg = self.inbound.get((node - direction * m, direction))
            if not reg:
                continue
            for h in reg.values():
                if h is run or h.spec.priority >= spec.priority:
                    continue
                j = h.idx + m
                if j >= len(h.spec.hops):
                    continue
                hh = h.spec.hops[j]
                if hh.kind != "section" or hh.sections[0] != sec:
                    continue
                ready = h.ready
                if ready is None:
                    continue
                for x in range(m):                       # estimate: fast running over the sections behind
                    hx = h.spec.hops[h.idx + x]
                    if hx.kind == "section":
                        ready += 0.95 * hx.shares[0]
                if ready < T - EPS or ready > T + self.mp.yield_window:
                    continue
                # would H lose time if `run` enters first?
                req = self.net.headway[sec]
                gap = self.mp.exit_gap
                pg = self.pairs.get((sec, direction))
                if pg:
                    req, gap = pg.get((h.spec.train, spec.train), (req, gap))
                h_entry = max(ready, T + req)
                d_h = max(0.0, ready - (h.t0 + h.spec.visits[j].sched_dep))
                h_run_time = hh.shares[0] - min(d_h, h.spec.slack * hh.shares[0])
                lost = (h_entry - ready) + max(0.0, (l_arr + gap) - (h_entry + h_run_time))
                if lost <= self.mp.yield_min_conflict:
                    continue
                if best is None or ready < best[1]:
                    best = (h, ready)
        return best

    def _traverse(self, run, i, T):
        spec = run.spec
        hop = spec.hops[i]
        v = spec.visits[i]
        t_in = T
        sched_start = run.t0 + v.sched_dep
        n = len(hop.sections)
        for j, (sec, ts) in enumerate(zip(hop.sections, hop.shares)):
            direction = hop.direction
            if j > 0:                                   # pass-through node inside a multi-section hop
                t_e, cause = self._entry_time(run, sec, direction, t_in)
                if t_e > t_in + EPS:
                    code, who, eid, root = cause
                    self.ledger.add(run, code, t_e - t_in, who, eid, root)
                    t_in = t_e
            self.last_entry[(sec, direction)] = (t_in, run)
            d_now = max(0.0, t_in - sched_start)
            rec = min(d_now, spec.slack * ts)
            w = self.delays.weather_factor(sec, t_in)
            w_extra = ts * (w - 1.0)
            x, eid = self.delays.incident(run, sec, t_in)
            arr_free = t_in + ts - rec + w_extra + x
            if rec > EPS:
                self.ledger.recover(run, rec)
            if w_extra > EPS:
                self.ledger.add(run, C.WEATHER, w_extra, "", 0, 0)
            if x > EPS:
                self.ledger.add(run, C.INCIDENT, x, "", eid, eid)
            arr = arr_free
            lead = self.last_arr.get((sec, direction))
            if lead is not None and lead[1] is not run:
                gap = self.mp.exit_gap
                pg = self.pairs.get((sec, direction))
                if pg:
                    gap = pg.get((spec.train, lead[1].spec.train), (0, gap))[1]
                floor = lead[0] + gap
                if floor > arr_free + EPS:
                    arr = floor
                    self.ledger.add(run, C.SECTION_WAIT, arr - arr_free, lead[1].spec.train, 0,
                                    self.ledger.root_of(lead[1]))
            self.last_arr[(sec, direction)] = (arr, run)
            sched_start += ts
            t_in = arr
        self._arrive(run, i + 1, t_in)

