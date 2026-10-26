"""Ground-truth delay ledger (SIMULATED).

Every minute of a train's current delay belongs to a component (cause code, causing train, event id,
root event id). Additions are recorded with their cause; recovery (slack, shortened halts) removes
minutes proportionally from all components, so the components always sum to the train's delay.
"""
from . import causes as C


class Ledger:
    def __init__(self, recorder=None):
        self.recorder = recorder
        self.state = {}                 # rid -> [comps, added, recovered_since_last_row]
        self.max_mismatch = 0.0         # largest |sum(components) - observed delay| seen (should stay ~0)
        self.checked = 0

    def _st(self, run):
        s = self.state.get(run.rid)
        if s is None:
            s = self.state[run.rid] = [{}, {}, 0.0]
        return s

    def add(self, run, code, minutes, cause_train="", event_id=0, root=0):
        if minutes <= 0:
            return
        comps, added, _ = self._st(run)
        key = (code, cause_train, event_id, root)
        comps[key] = comps.get(key, 0.0) + minutes
        added[key] = added.get(key, 0.0) + minutes

    def recover(self, run, minutes):
        st = self._st(run)
        comps, added = st[0], st[1]
        total = sum(comps.values())
        if total <= 0 or minutes <= 0:
            return
        f = min(1.0, minutes / total)
        for k in list(comps):
            dec = comps[k] * f
            comps[k] -= dec
            if k in added:
                added[k] = max(0.0, added[k] - dec)
        st[2] += min(minutes, total)

    def root_of(self, run):
        """Root event id of the largest component of this train's current delay (0 if none)."""
        s = self.state.get(run.rid)
        if not s or not s[0]:
            return 0
        key = max(s[0], key=s[0].get)
        return key[3] if s[0][key] > 1e-9 else 0

    def total(self, run):
        s = self.state.get(run.rid)
        return sum(s[0].values()) if s else 0.0

    def obs(self, run, i, phase):
        """Called by the engine at each arrival and departure commit."""
        st = self._st(run)
        v = run.spec.visits[i]
        t = run.arr[i] if phase == "arr" else run.dep[i]
        sched = run.t0 + (v.sched_arr if phase == "arr" else v.sched_dep)
        self.checked += 1
        mism = abs(sum(st[0].values()) - (t - sched))
        if mism > self.max_mismatch:
            self.max_mismatch = mism
        if self.recorder is not None:
            comps = [(k, m) for k, m in st[0].items() if m > 0.01]
            self.recorder.visit_obs(run, i, phase, t, comps, dict(st[1]) if phase == "arr" else None, st[2])
        if phase == "arr":
            st[1].clear()
            st[2] = 0.0

    def finish(self, run):
        self.state.pop(run.rid, None)
