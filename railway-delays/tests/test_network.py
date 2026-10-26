"""Tests for the network and spec builder on the real Phase 2 tables."""
import pytest

pytest.importorskip("pandas")
from pathlib import Path

if not (Path("data/processed/trains.parquet").exists() and Path("data/raw/trains.json").exists()):
    pytest.skip("Phase 2 outputs not present", allow_module_level=True)

from src.simulation.network import load_specs


@pytest.fixture(scope="module")
def built():
    return load_specs()


def test_prayagraj_is_one_node(built):
    net, specs, stats, cfg = built
    assert net.node_names.count("PRYG") == 1
    assert not {"ALD", "NYN", "COI"} & set(net.node_names)
    assert len(net.node_names) == 108 and net.n_sections == 107


def test_slips_are_dropped_and_legs_found(built):
    net, specs, stats, cfg = built
    assert len(specs) == 225 and not any("Slip" in t for t in specs)
    legs = [t for t, s in specs.items() if any(h.kind == "leg" for h in s.hops)]
    assert len(legs) == 39


def test_every_section_hop_has_nonnegative_scheduled_time(built):
    net, specs, stats, cfg = built
    for s in specs.values():
        for h in s.hops:
            if h.kind == "section":
                assert h.shares and all(x >= 0 for x in h.shares)


def test_scheduled_time_between_halts_is_conserved(built):
    net, specs, stats, cfg = built
    s = specs["12302"]                     # Rajdhani with few halts
    anchors = [k for k, v in enumerate(s.visits) if v.halt or v.sched_dep - v.sched_arr > 0]
    for a, b in zip(anchors, anchors[1:]):
        if s.hops[a].kind == "leg":
            continue
        total = sum(sum(s.hops[k].shares) for k in range(a, b))
        assert total == pytest.approx(s.visits[b].sched_arr - s.visits[a].sched_dep, abs=1e-6)


def test_no_train_reverses_after_merging_prayagraj(built):
    net, specs, stats, cfg = built
    for s in specs.values():
        d = [h.direction for h in s.hops if h.kind == "section"]
        assert all(x == d[0] for x in d)


def test_link_statistics_add_up(built):
    net, specs, stats, cfg = built
    assert stats["linked"] == stats["return_train"] + stats["odd_even"]
    assert stats["linked"] + stats["no_partner"] + stats["turnaround_out_of_window"] == stats["trains"]
    assert sum(1 for s in specs.values() if s.feeder) == stats["linked"]
