"""Unit tests for the core movement rules: headway, priority, overtaking, slack, halts, rake reuse."""
import pytest

from src.simulation.engine import MoveParams, rake_delay
from tests.helpers import FixedEntry, delays_at, line_spec, make_net, run, sim


def test_headway_between_trains_entering_the_same_section():
    net = make_net(4, headway=5.0)
    s = sim(net)
    a = run(s, line_spec("A", 3, 100, 10))
    b = run(s, line_spec("B", 3, 100, 10))
    s.drain()
    # equal priority, same start: B enters section 0 five minutes after A
    assert a.dep[0] == 100
    assert b.dep[0] == pytest.approx(105.0)
    # and keeps at least the headway on every later section
    for i in range(3):
        assert b.dep[i] - a.dep[i] >= 5.0 - 1e-9


def test_headway_is_configurable_per_section():
    net = make_net(3, headway=5.0)
    net.headway[1] = 9.0
    s = sim(net)
    a = run(s, line_spec("A", 3, 100, 10, last=2))
    b = run(s, line_spec("B", 3, 100, 10, last=2))
    s.drain()
    assert b.dep[1] - a.dep[1] >= 9.0 - 1e-9


def test_follower_cannot_pass_inside_a_section():
    net = make_net(3, headway=5.0)
    s = sim(net, MoveParams(exit_gap=2.0))
    slow = run(s, line_spec("slow", 5, 100, 30, last=2, typ="Passenger"))
    fast = run(s, line_spec("fast", 1, 105, 10, last=2, typ="Rajdhani"))
    s.drain()
    # fast would arrive at 115 but must stay behind slow (arrives 130) by the exit gap
    assert slow.arr[1] == pytest.approx(130.0)
    assert fast.arr[1] == pytest.approx(132.0)


def test_lower_priority_yields_at_a_junction():
    net = make_net(4, junctions={1})
    s = sim(net, MoveParams(yield_window=10, max_yield_hold=30))
    low = run(s, line_spec("L", 5, 40, [10, 30, 30], typ="Passenger"))   # slow after node 1; passes node 1 at 50
    high = run(s, line_spec("H", 1, 45, 10, typ="Rajdhani"))      # ready at node 1 at 55
    s.drain()
    assert high.dep[1] == pytest.approx(55.0)                      # priority train is not delayed
    assert low.dep[1] == pytest.approx(60.0)                       # waits until H entered + headway
    assert low.dep[1] - high.dep[1] >= 5.0 - 1e-9


def test_no_yield_at_a_plain_station_without_halt():
    net = make_net(4, junctions=set())
    s = sim(net)
    low = run(s, line_spec("L", 5, 40, [10, 30, 30], typ="Passenger"))
    high = run(s, line_spec("H", 1, 45, 10, typ="Rajdhani"))
    s.drain()
    # L passes node 1 without stopping and cannot be overtaken there: it keeps its slot
    assert low.dep[1] == pytest.approx(50.0)
    assert high.dep[1] >= 55.0 - 1e-9


def test_overtaking_allowed_where_slower_train_has_a_real_halt():
    net = make_net(4, junctions=set())
    s = sim(net)
    low = run(s, line_spec("L", 5, 40, [10, 30, 30], halts={1: 3.0}, typ="Passenger"))   # at node 1 from 50 to 53
    high = run(s, line_spec("H", 1, 45, 10, typ="Rajdhani"))
    s.drain()
    assert high.dep[1] == pytest.approx(55.0)                      # passes the halted train, no delay
    assert low.dep[1] == pytest.approx(60.0)                       # held so the express goes first


def test_yield_hold_is_capped():
    net = make_net(4, junctions={1})
    s = sim(net, MoveParams(max_yield_hold=3.0))
    low = run(s, line_spec("L", 5, 40, [10, 30, 30], typ="Passenger"))
    run(s, line_spec("H", 1, 45, 10, typ="Rajdhani"))
    s.drain()
    assert low.yield_held <= 3.0 + 1e-9


def test_lookback_depth_two_sees_a_train_two_sections_behind():
    def dep_at_junction(depth):
        net = make_net(5, junctions={2})
        s = sim(net, MoveParams(lookback=depth, yield_window=15))
        low = run(s, line_spec("L", 5, 30, [10, 10, 40, 40], last=4, typ="Passenger"))   # leaves node 2 at 50, slow after
        # H reaches node 1 at 55 and node 2 at about 64.5 (95% of the scheduled section time)
        run(s, line_spec("H", 1, 45, 10, last=4, typ="Rajdhani"))
        s.drain()
        return low.dep[2]
    assert dep_at_junction(1) == pytest.approx(50.0)     # depth 1 cannot see H yet
    assert dep_at_junction(2) > 50.0                     # depth 2 holds L for H


def test_slack_recovery_is_limited_to_slack_fraction_of_the_section():
    net = make_net(5)
    s = sim(net, delays=FixedEntry(20.0))
    r = run(s, line_spec("T", 3, 0, 100, last=4, slack=0.06, entered=True))
    s.drain()
    arr = [d[0] for d in delays_at(r)]
    # 20 min late, recovers at most 6 min (6% of 100) per section, never below zero
    assert arr == pytest.approx([20.0, 14.0, 8.0, 2.0, 0.0])


def test_late_train_shortens_a_halt_to_its_minimum():
    net = make_net(3)
    s = sim(net, delays=FixedEntry(20.0))
    spec = line_spec("T", 3, 100, 50, last=2, halts={1: 10.0}, entered=True)
    r = run(s, spec)
    s.drain()
    # entry 20 min late; the first section recovers 6% of 50 = 3 min, so arrival is 17 min late.
    # Scheduled halt 10 min, minimum 5: the halt shrinks by 5, leaving 12 min late.
    arr1, dep1 = delays_at(r)[1]
    assert arr1 == pytest.approx(17.0)
    assert dep1 == pytest.approx(12.0)


def test_on_time_train_keeps_its_scheduled_halt():
    net = make_net(3)
    s = sim(net)
    r = run(s, line_spec("T", 3, 100, 50, last=2, halts={1: 10.0}))
    s.drain()
    assert delays_at(r) == [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]


def test_off_corridor_leg_carries_delay_and_does_not_block():
    from src.simulation.engine import Hop, TrainSpec, Visit
    net = make_net(3)
    leg = TrainSpec("LEG", "Superfast", 3, 0.06,
                    [Visit(0, 0, 0, True), Visit(2, 300, 300, True)], [Hop("leg", leg_minutes=300.0)],
                    entered_mid_route=True)
    s = sim(net, delays=FixedEntry(12.0))
    r = run(s, leg)
    other = run(s, line_spec("O", 3, 0, 10, last=2))
    s.drain()
    assert r.arr[1] == pytest.approx(300.0 + 12.0)                 # delay carried, no recovery
    assert other.dep[0] == 0.0                                    # the leg used no section: no headway wait


class TestRake:
    def test_delay_is_absorbed_by_the_turnaround_buffer(self):
        assert rake_delay(20, gap_sched=45 + 30, min_turnaround=45, feeder_exits_mid=False,
                          train_entered_mid=False, recovery=0.3) == 0.0

    def test_excess_delay_is_passed_on(self):
        assert rake_delay(40, gap_sched=45 + 10, min_turnaround=45, feeder_exits_mid=False,
                          train_entered_mid=False, recovery=0.3) == pytest.approx(30.0)

    def test_off_corridor_recovery_applies(self):
        d = rake_delay(100, 45, 45, True, True, 0.3)
        assert d == pytest.approx(100 * 0.7 * 0.7)

    def test_engine_passes_feeder_delay_to_the_next_service(self):
        net = make_net(3)
        s = sim(net, MoveParams(min_turnaround=45.0))
        a = run(s, line_spec("A", 3, 0, 30, last=2, entered=False), day=0)
        a.spec.entered_mid_route = False
        # A is late by 25 at its terminus: use an entry delay on A via a delays stub
        s2 = sim(net, MoveParams(min_turnaround=45.0), delays=FixedEntry(25.0))
        a = run(s2, line_spec("A", 3, 0, 30, last=2, entered=True), day=0)
        b_spec = line_spec("B", 3, 130, 30, last=0, feeder=("A", 0, 45 + 10))      # buffer 10 min
        b = run(s2, b_spec, day=0)
        s2.drain()
        # A arrives 60 + 25 recovered some slack: exit delay is A's delay at its last node
        assert b.entry_delay == pytest.approx(max(0.0, a.exit_delay - 10.0))
        assert b.entry_delay > 0


def test_no_yield_when_going_first_does_not_delay_the_priority_train():
    net = make_net(4, junctions={1})
    s = sim(net, MoveParams(yield_window=10))
    low = run(s, line_spec("L", 5, 40, 10, typ="Passenger"))       # same speed as H, 5 min ahead
    high = run(s, line_spec("H", 1, 45, 10, typ="Rajdhani"))
    s.drain()
    assert low.dep[1] == pytest.approx(50.0) and low.yield_held == 0.0
    assert high.dep[1] == pytest.approx(55.0)


# ---- any-station overtaking with limited loop capacity -------------------------------------------
def _any(**kw):
    return MoveParams(overtake_mode="any_station", yield_window=10, **kw)


def test_any_station_mode_lets_a_priority_train_overtake_at_a_plain_station():
    net = make_net(4, junctions=set())
    s = sim(net, _any(loop_capacity=2, overtake_penalty=0.0))
    low = run(s, line_spec("L", 5, 40, [10, 30, 30], typ="Passenger"))       # no halt, no junction at node 1
    high = run(s, line_spec("H", 1, 45, 10, typ="Rajdhani"))
    s.drain()
    assert high.dep[1] == pytest.approx(55.0)
    assert low.dep[1] == pytest.approx(60.0)


def test_old_rule_is_kept_as_an_option():
    net = make_net(4, junctions=set())
    s = sim(net, MoveParams(overtake_mode="junction_or_halt", yield_window=10))
    low = run(s, line_spec("L", 5, 40, [10, 30, 30], typ="Passenger"))
    run(s, line_spec("H", 1, 45, 10, typ="Rajdhani"))
    s.drain()
    assert low.dep[1] == pytest.approx(50.0) and low.yield_held == 0.0


def test_overtaken_train_pays_the_configured_penalty():
    net = make_net(4, junctions=set())
    s = sim(net, _any(loop_capacity=2, overtake_penalty=3.0))
    low = run(s, line_spec("L", 5, 40, [10, 30, 30], typ="Passenger"))
    run(s, line_spec("H", 1, 45, 10, typ="Rajdhani"))
    s.drain()
    assert low.dep[1] == pytest.approx(63.0)               # priority train enters at 55, +5 headway, +3 penalty


def test_loop_capacity_limits_how_many_trains_are_held_aside():
    def yielded(cap):
        net = make_net(3, junctions=set())                 # the only station where a train can wait is node 1
        s = sim(net, _any(loop_capacity=cap, overtake_penalty=0.0))
        l1 = run(s, line_spec("L1", 5, 50, [30], first=1, last=2, typ="Passenger"))   # start at node 1
        l2 = run(s, line_spec("L2", 5, 51, [30], first=1, last=2, typ="Passenger"))
        run(s, line_spec("H", 1, 42, [10, 10], last=2, typ="Rajdhani"))              # reaches node 1 at 52
        s.drain()
        return l1.yield_held > 0, l2.yield_held > 0
    assert yielded(0) == (False, False)
    assert yielded(1) == (True, False)           # only one train can wait in the loop
    assert yielded(2) == (True, True)


def test_held_count_returns_to_zero():
    net = make_net(4, junctions=set())
    s = sim(net, _any(loop_capacity=1, overtake_penalty=0.0))
    run(s, line_spec("L", 5, 40, [10, 30, 30], typ="Passenger"))
    run(s, line_spec("H", 1, 45, 10, typ="Rajdhani"))
    s.drain()
    assert all(v == 0 for v in s.held.values())
