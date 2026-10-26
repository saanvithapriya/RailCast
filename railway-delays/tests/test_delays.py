"""Tests for weather and primary delay sources (all SIMULATED)."""
import random
from collections import Counter

import pytest

from src.simulation import causes as C
from src.simulation.config import load_config
from src.simulation.delays import DelayModel
from src.simulation.engine import MoveParams, Net, Simulator
from src.simulation.weather import FixedWeather, SyntheticWeather, WeatherSource, day_to_date
from tests.helpers import delays_at, line_spec, run


def small_net(n=4, headway=5.0):
    names = ["NDLS", "A", "CNB", "PRYG", "B", "C"][:n]
    return Net(names, [False] * n, [10.0] * (n - 1), [headway] * (n - 1), [1.0] * (n - 1), [False] * (n - 1),
               [1.0] * (n - 1))


class FakeRun:
    def __init__(self, seed=1, typ="Superfast", prio=3, jf=0.0):
        self.rnd = random.Random(seed)
        self.spec = type("S", (), {"train": "T", "type": typ, "priority": prio, "extra": {"journey_fraction": jf}})()


def test_fog_is_concentrated_in_dec_jan_and_absent_in_summer():
    w = SyntheticWeather(load_config(), seed=42)
    by_month = Counter()
    for d in range(365):
        if w.state(d) == "fog":
            by_month[day_to_date(d).month] += 1
    assert by_month[1] + by_month[12] > 0.6 * sum(by_month.values())
    assert sum(by_month[m] for m in range(4, 10)) == 0
    assert by_month[1] > 5


def test_weather_is_reproducible_and_independent_of_the_range_requested():
    a, b = SyntheticWeather(load_config(), 42), SyntheticWeather(load_config(), 42)
    assert [a.state(d) for d in range(300)] == [b.state(d) for d in range(300)]
    c = SyntheticWeather(load_config(), 42)
    assert c.state(300) == a.state(300)               # asking for a late day first gives the same answer


def test_weather_source_is_swappable():
    class AlwaysFog(WeatherSource):
        source = "test"

        def state(self, day):
            return "fog"
    cfg = load_config()
    net = small_net()
    m = DelayModel(net, cfg, AlwaysFog(), 1)
    assert m.weather_factor(0, 100.0) == pytest.approx(1 + cfg["weather"]["fog_slowdown"])


def test_fog_slows_a_train_down_through_the_engine():
    cfg = load_config()
    net = small_net()
    clear = Simulator(net, MoveParams(), DelayModel(net, cfg, FixedWeather("clear"), 1))
    fog = Simulator(net, MoveParams(), DelayModel(net, cfg, FixedWeather("fog"), 1))
    c = run(clear, line_spec("T", 3, 100, 20))
    f = run(fog, line_spec("T", 3, 100, 20))
    clear.drain()
    fog.drain()
    assert f.arr[-1] > c.arr[-1] + 3.0


def test_fog_is_weaker_far_from_delhi():
    cfg = load_config()
    net = small_net()
    net.section_fog = [1.0, 0.5, 0.2]
    m = DelayModel(net, cfg, FixedWeather("fog"), 1)
    assert m.weather_factor(0, 0) > m.weather_factor(1, 0) > m.weather_factor(2, 0) > 1.0


def test_incident_probability_scales_with_factors_and_has_a_heavy_tail():
    cfg = load_config()
    cfg["incidents"]["base_rate"] = 0.2
    net = small_net()
    m = DelayModel(net, cfg, FixedWeather("clear"), 1)
    mins = [m.incident(FakeRun(i), 0, 12 * 60.0)[0] for i in range(4000)]      # off-peak, January
    hit = sorted(x for x in mins if x > 0)
    assert 0.2 * 1.5 * 0.7 < len(hit) / 4000 < 0.2 * 1.5 * 1.3                  # month factor 1.5 in January
    assert hit[-1] > 3 * hit[len(hit) // 2]                                     # heavy tail
    assert max(hit) <= cfg["incidents"]["cap_min"]
    peak = sum(1 for i in range(4000) if m.incident(FakeRun(i), 0, 7 * 60.0)[0] > 0)
    assert peak > len(hit)                                                      # rush hour has more incidents
    fog = DelayModel(net, cfg, FixedWeather("fog"), 1)
    nfog = sum(1 for i in range(4000) if fog.incident(FakeRun(i), 0, 12 * 60.0)[0] > 0)
    assert nfog > len(hit) * 1.5                                                # fog doubles incident probability


def test_incident_is_added_as_running_time_and_logged():
    cfg = load_config()
    cfg["incidents"]["base_rate"] = 1.0                                         # every section has an incident
    net = small_net()
    dm = DelayModel(net, cfg, FixedWeather("clear"), 1)
    s = Simulator(net, MoveParams(), dm)
    r = run(s, line_spec("T", 3, 100, 20))
    s.drain()
    assert delays_at(r)[-1][0] > 0
    assert len(dm.events) == 3 and all(e[2] == 1 and e[7] == "T" for e in dm.events)


def test_entry_delay_zero_share_and_fog_effect():
    cfg = load_config()
    net = small_net()
    clear = DelayModel(net, cfg, FixedWeather("clear"), 1)
    fog = DelayModel(net, cfg, FixedWeather("fog"), 1)
    c = [clear.entry_delay(FakeRun(i), 100.0)[0] for i in range(3000)]
    f = [fog.entry_delay(FakeRun(i), 100.0)[0] for i in range(3000)]
    z = cfg["entry_delay"]["zero_probability"]
    assert z - 0.1 < sum(1 for x in c if x == 0) / len(c) < z + 0.1
    assert sum(f) > 1.8 * sum(c)


def test_freight_blocks_a_train_but_not_a_rajdhani():
    cfg = load_config()
    net = small_net()
    dm = DelayModel(net, cfg, FixedWeather("clear"), 1)
    dm.blocks[0].append((95.0, 140.0, 99, C.FREIGHT, 0))
    s = Simulator(net, MoveParams(), dm)
    slow = run(s, line_spec("S", 4, 100, 20, typ="Mail/Express"))
    fast = run(s, line_spec("R", 1, 100, 20, typ="Rajdhani"))
    s.drain()
    assert slow.dep[0] == pytest.approx(140.0) or slow.dep[0] >= 140.0          # waits for the freight to clear
    assert fast.dep[0] == pytest.approx(100.0)                                  # freight yields to a Rajdhani


def test_disruption_is_directional_for_signal_failure_and_both_ways_for_a_block():
    cfg = load_config()
    net = small_net()
    dm = DelayModel(net, cfg, FixedWeather("clear"), 1)
    dm.blocks[1].append((0.0, 100.0, 7, C.DISRUPTION, 1))                       # signal failure, down line only
    dm.blocks[1].append((0.0, 50.0, 8, C.DISRUPTION, 0))                        # block, both lines
    assert dm.block_end(1, 1, 10.0, 1) == (100.0, C.DISRUPTION, 7)              # blocks even a Rajdhani
    assert dm.block_end(1, -1, 10.0, 3) == (50.0, C.DISRUPTION, 8)
    assert dm.block_end(1, 1, 120.0, 3) is None


def test_generated_blockers_exist_and_are_reproducible():
    cfg = load_config()
    net = small_net(6)
    a, b = DelayModel(net, cfg, FixedWeather(), 42), DelayModel(net, cfg, FixedWeather(), 42)
    for d in range(10, 20):
        a.generate_day(d)
        b.generate_day(d)
    assert a.events == b.events and len(a.events) > 10
    assert {e[2] for e in a.events} <= {4, 5, 6}


def test_disruptions_last_between_30_and_180_minutes():
    cfg = load_config()
    cfg["disruptions"]["rate_per_section_day"] = 3.0
    net = small_net(6)
    m = DelayModel(net, cfg, FixedWeather(), 1)
    for d in range(5):
        m.generate_day(d)
    durs = [e[6] for e in m.events if e[2] in (4, 5)]
    assert durs and 30 <= min(durs) and max(durs) <= 180
