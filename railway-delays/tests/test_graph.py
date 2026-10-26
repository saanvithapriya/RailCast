"""Phase 3 graph tests. Delay-based parts use small hand-made SIMULATED-style data with known values."""
import builtins
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

if not (Path("data/processed/trains.parquet").exists() and Path("data/raw/trains.json").exists()):
    pytest.skip("Phase 2 outputs not present", allow_module_level=True)

from src.graph import delay_attrs, snapshot as snap_mod
from src.graph import station_graph as sg
from src.graph import train_graph as tg
from src.graph.groundtruth import load_groundtruth
from src.graph.timetable import load_timetable


@pytest.fixture(scope="module")
def tt():
    return load_timetable()


@pytest.fixture(scope="module")
def tables(tt):
    return sg.build_station_tables(tt)


# ------------------------------------------------------------------ station graph
def test_node_and_edge_counts(tt, tables):
    nodes, edges, node_map = tables
    assert len(nodes) == 108 and len(edges) == 214 and len(node_map) == 110
    assert edges.edge_id.is_unique and nodes.node.is_unique


def test_prayagraj_is_one_node_with_mapping_back(tables):
    nodes, edges, node_map = tables
    pr = nodes[nodes.node_name == "PRYG"].iloc[0]
    assert pr.original_codes == "ALD|NYN|COI" and pr.n_original_stations == 3
    m = node_map[node_map.node == pr.node]
    assert set(m.station_code) == {"ALD", "NYN", "COI"}
    assert not {"ALD", "NYN", "COI"} & set(nodes.node_name)


def test_edges_are_along_the_chain_only_and_both_directions(tables):
    nodes, edges, _ = tables
    assert ((edges.from_node - edges.to_node).abs() == 1).all()
    fwd = set(zip(edges[edges.direction == 1].from_node, edges[edges.direction == 1].to_node))
    bwd = set(zip(edges[edges.direction == -1].from_node, edges[edges.direction == -1].to_node))
    assert {(b, a) for a, b in fwd} == bwd and len(fwd) == 107
    g = sg.build_station_graph(nodes, edges)
    assert nx.is_strongly_connected(g)


def test_hourly_counts_add_up_and_shares_are_valid(tables):
    nodes, edges, _ = tables
    hours = edges[[f"trains_hour_{h:02d}" for h in range(24)]].sum(axis=1)
    assert (hours == edges.trains_per_day).all()
    assert (edges.n_fast + edges.n_slow == edges.trains_per_day).all()
    share_cols = [c for c in nodes.columns if c.startswith("share_")]
    assert np.allclose(nodes[share_cols].sum(axis=1), 1.0)
    assert (nodes.halts_per_day <= nodes.trains_per_day * 2).all()


def test_neighbours_of_kanpur_query(tt, tables):
    nodes, edges, _ = tables
    g = sg.build_station_graph(nodes, edges)
    cnb = int(nodes[nodes.node_name == "CNB"].node.iloc[0])
    nb = {(g.nodes[n]["node_name"], d["trains_per_day"]) for _, n, d in g.out_edges(cnb, data=True)}
    assert len(nb) == 2 and all(t_ > 0 for _, t_ in nb)
    assert {g.nodes[n]["node_name"] for n in g.predecessors(cnb)} == {g.nodes[n]["node_name"] for n in g.successors(cnb)}


# ------------------------------------------------------------------ train graph
def test_train_graph_nodes_and_rake_links(tt):
    nodes, edges = tg.build_train_tables(tt)
    assert len(nodes) == 225
    rake = edges[edges.type == "rake"]
    assert len(rake) == 218 and set(rake.rule) == {"return_train", "odd_even"}
    assert (rake.rule == "return_train").sum() == 214
    assert set(edges.type) == {"rake", "section_sharing", "overtake_station", "overtake_section"}
    assert edges.source.isin(nodes.train).all() and edges.target.isin(nodes.train).all()
    assert (edges["count"] >= 1).all()
    g = tg.build_train_graph(nodes, edges)
    assert g.number_of_nodes() == 225


def test_section_sharing_gap_is_below_the_window(tt):
    edges = tg.section_sharing_edges(tt, window_min=10.0)
    assert (edges.mean_gap_min < 10.0).all() and (edges.mean_gap_min >= 0).all()
    assert (edges.source < edges.target).all()          # stored once, undirected


# ------------------------------------------------------------------ leakage rules
def test_timetable_graphs_never_touch_simulation_outputs(monkeypatch):
    """Build the timetable-only graphs while any access to a simulation output path raises."""
    real_open, real_read = builtins.open, pd.read_parquet

    def guard(path):
        s = str(path).replace("\\", "/")
        if "/sim/" in s or "sim_pilot" in s or "/sim" in s.split("processed")[-1][:6]:
            raise AssertionError(f"timetable-only code read a simulation output: {s}")

    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (guard(f), real_open(f, *a, **k))[1])
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: (guard(p), real_read(p, *a, **k))[1])
    tt2 = load_timetable()
    nodes, edges, _ = sg.build_station_tables(tt2)
    tg.build_train_tables(tt2)
    assert len(nodes) == 108


def test_timetable_modules_do_not_import_the_delay_modules():
    for f in ("timetable.py", "station_graph.py", "train_graph.py"):
        lines = [ln for ln in Path("src/graph", f).read_text().splitlines() if ln.lstrip().startswith(("import ", "from "))]
        joined = "\n".join(lines)
        assert "delay_attrs" not in joined and "snapshot" not in joined and "groundtruth" not in joined, f
        assert "src.simulation.engine" not in joined, f
        assert "processed/sim" not in Path("src/graph", f).read_text(), f


def test_groundtruth_loader_refuses_without_the_validation_flag():
    with pytest.raises(PermissionError):
        load_groundtruth("data/processed/sim/medium")


# ------------------------------------------------------------------ date-range functions (hand-made data)
def _write_scenario(root, rows, queues=None):
    """rows: list of dicts with train, date (YYYY-MM-DD run date), seq, station, arr, dep, arr_delay, dep_delay."""
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df.date)
    out = pd.DataFrame({
        "date": df.date, "train": pd.Categorical(df.train), "station": pd.Categorical(df.station), "seq": df.seq.astype("int16"),
        "scheduled_arrival": np.float32(0), "scheduled_departure": np.float32(0),
        "actual_arrival": df.arr.astype("float32"), "actual_departure": df.dep.astype("float32"),
        "arrival_delay": df.arr_delay.astype("float32"), "departure_delay": df.dep_delay.astype("float32"),
    })
    for m, g in out.groupby(out.date.dt.strftime("%Y-%m")):
        p = Path(root) / "observations" / f"month={m}"
        p.mkdir(parents=True, exist_ok=True)
        g.to_parquet(p / "part.parquet", index=False)
    q = pd.DataFrame(queues or [], columns=["section", "direction", "wait_start", "wait_end", "train"])
    q.to_parquet(Path(root) / "queues.parquet", index=False)


@pytest.fixture()
def hand_scenario(tmp_path, tt):
    # two runs of train T from NDLS (node 0) to CSB (node 1): day 2025-03-31 (late evening, spills into April) and 2025-04-01
    rows = []
    for d, base in (("2025-03-31", 23 * 60 + 50), ("2025-04-01", 100)):
        rows.append(dict(train="T", date=d, seq=0, station="NDLS", arr=np.nan, dep=base, arr_delay=np.nan, dep_delay=2.0))
        rows.append(dict(train="T", date=d, seq=1, station="CSB", arr=base + 20, dep=base + 21, arr_delay=12.0 if d == "2025-03-31" else 30.0, dep_delay=12.0))
    queues = [(0, 1, 1.0, 5.0, "T")]
    _write_scenario(tmp_path, rows, queues)
    return tmp_path


def test_window_includes_a_spilling_run_only_by_event_time(hand_scenario, tt):
    # run 1 enters 2025-03-31 23:50 and arrives at CSB at 2025-04-01 00:10 -> belongs to April, not March
    mar = delay_attrs.edge_delay_attributes(hand_scenario, "2025-03-01", "2025-03-31", tt.node_of)
    apr = delay_attrs.edge_delay_attributes(hand_scenario, "2025-04-01", "2025-04-30", tt.node_of)
    assert len(mar) == 0
    assert len(apr) == 1 and apr.n_passages.iloc[0] == 2
    assert apr.mean_arrival_delay.iloc[0] == pytest.approx(21.0)          # (12 + 30) / 2
    assert apr.mean_delay_added.iloc[0] == pytest.approx(((12 - 2) + (30 - 2)) / 2)


def test_later_data_cannot_change_an_earlier_window(hand_scenario, tt):
    a = delay_attrs.edge_delay_attributes(hand_scenario, "2025-04-01", "2025-04-01", tt.node_of)
    # tamper with everything after the window: change the second run (arrives 2025-04-01 01:50 and is inside)
    # then add a run on 2025-04-05; the 04-01 window must still contain exactly the same passages
    extra = pd.read_parquet(hand_scenario / "observations" / "month=2025-04" / "part.parquet")
    row = extra.iloc[[0, 1]].copy()
    row["date"] = pd.Timestamp("2025-04-05")
    row["arrival_delay"] = np.float32(500)
    pd.concat([extra, row], ignore_index=True).to_parquet(hand_scenario / "observations" / "month=2025-04" / "part.parquet", index=False)
    b = delay_attrs.edge_delay_attributes(hand_scenario, "2025-04-01", "2025-04-01", tt.node_of)
    pd.testing.assert_frame_equal(a, b)
    c = delay_attrs.edge_delay_attributes(hand_scenario, "2025-04-01", "2025-04-30", tt.node_of)
    assert c.n_passages.iloc[0] == 3


def test_queue_waits_follow_the_window(hand_scenario, tt):
    # queue wait ended at absolute minute 5 of 2025-01-01: inside January, outside April
    jan = delay_attrs.edge_delay_attributes(hand_scenario, "2025-01-01", "2025-04-30", tt.node_of)
    assert jan.n_waits.iloc[0] == 1 and jan.wait_minutes.iloc[0] == pytest.approx(4.0)
    apr = delay_attrs.edge_delay_attributes(hand_scenario, "2025-04-01", "2025-04-30", tt.node_of)
    assert apr.n_waits.iloc[0] == 0


def test_node_delay_attributes_respect_bounds(hand_scenario, tt):
    nd = delay_attrs.node_delay_attributes(hand_scenario, "2025-04-01", "2025-04-30", tt.node_of)
    csb = nd[nd.node == tt.node_of["CSB"]].iloc[0]
    assert csb.n_arrivals == 2 and csb.mean_arrival_delay == pytest.approx(21.0)


# ------------------------------------------------------------------ snapshot
def test_snapshot_returns_the_right_trains(hand_scenario, tt):
    rows = snap_mod.load_runs_around(hand_scenario, delay_attrs.to_day("2025-04-01") * 1440.0 + 100)
    day = delay_attrs.to_day("2025-04-01") * 1440.0
    # t = 04-01 00:05: run 1 (entered 03-31 23:50, left NDLS 23:50, arrives CSB 00:10) is in the section towards CSB
    s = snap_mod.positions_at(rows, day + 5, tt.node_of)
    assert len(s) == 1
    r = s.iloc[0]
    assert r.train == "T" and r.state == "in_section" and r.node == tt.node_of["CSB"] and r.from_node == tt.node_of["NDLS"]
    assert r.current_delay_min == pytest.approx(2.0)                       # departure delay at NDLS
    # t = 04-01 00:10:30: run 1 is at CSB (arrived 00:10, leaves 00:11)
    s = snap_mod.positions_at(rows, day + 10.5, tt.node_of)
    assert len(s) == 1 and s.iloc[0].state == "at_station" and s.iloc[0].station == "CSB"
    assert s.iloc[0].current_delay_min == pytest.approx(12.0)              # arrival delay
    # t = 04-01 12:00: nothing in the corridor
    assert len(snap_mod.positions_at(rows, day + 12 * 60, tt.node_of)) == 0
    # the snapshot uses nothing after t: removing every row that starts after t does not change it
    early = rows[rows.abs_act_dep <= day + 10.5]
    s2 = snap_mod.positions_at(early, day + 5, tt.node_of)
    assert len(s2) == 0 or s2.iloc[0].current_delay_min == pytest.approx(2.0)


def test_snapshot_wrapper_adds_names_and_type(hand_scenario, tt):
    s = snap_mod.snapshot(hand_scenario, "2025-04-01", 0, tt, minute=5)
    assert list(s.node_name) == ["CSB"] and s.is_simulated.all()


# ------------------------------------------------------------------ split config and multi-source windows
def test_split_config_is_the_single_source_of_the_windows():
    from src import splits
    w = splits.windows()
    assert w["train"] == ("2024-01-01", "2025-08-31")
    assert w["validation"] == ("2025-09-01", "2025-10-31") and w["test"] == ("2025-11-01", "2025-12-31")
    assert w["analysis_all_2025"] == ("2025-01-01", "2025-12-31")
    d = lambda s: pd.Timestamp(s)
    assert d(w["train"][1]) + pd.Timedelta(days=1) == d(w["validation"][0])        # contiguous, by time, no overlap
    assert d(w["validation"][1]) + pd.Timedelta(days=1) == d(w["test"][0])
    assert [s.start for s in splits.sources("medium")] == ["2024-01-01", "2025-01-01"]


def test_build_uses_the_config_windows_and_marks_the_analysis_version():
    from src import splits
    from src.graph import build
    assert build.tag("train") == "SIMULATED_train_2024-01-01_2025-08-31"
    assert build.tag("all2025") == "SIMULATED_ANALYSIS_ONLY_all_2025"
    src = Path("src/graph/build.py").read_text()
    assert "2024-01-01" not in src and "2025-08-31" not in src          # no hard-coded dates: read from the config


def _two_year_sources(tmp_path):
    from src.splits import Source
    d24, d25 = tmp_path / "s2024", tmp_path / "s2025"
    # 2024 run entering 2024-12-31 23:50 reaches CSB at 2025-01-01 00:10 (spills over the year boundary);
    # 2024 run on 2024-12-30 is entirely inside 2024
    rows24 = []
    for d, base in (("2024-12-30", 600), ("2024-12-31", 23 * 60 + 50)):
        rows24.append(dict(train="T", date=d, seq=0, station="NDLS", arr=np.nan, dep=base, arr_delay=np.nan, dep_delay=1.0))
        rows24.append(dict(train="T", date=d, seq=1, station="CSB", arr=base + 20, dep=base + 21, arr_delay=10.0, dep_delay=10.0))
    rows25 = []
    for d, base in (("2025-01-02", 300), ("2025-09-05", 300)):
        rows25.append(dict(train="T", date=d, seq=0, station="NDLS", arr=np.nan, dep=base, arr_delay=np.nan, dep_delay=1.0))
        rows25.append(dict(train="T", date=d, seq=1, station="CSB", arr=base + 20, dep=base + 21, arr_delay=50.0, dep_delay=50.0))
    _write_scenario(d24, rows24)
    _write_scenario(d25, rows25)
    return [Source(str(d24), "2024-01-01", "2024-12-31"), Source(str(d25), "2025-01-01", "2025-12-31")]


def test_window_spans_two_sources_without_double_counting_the_year_boundary(tmp_path, tt):
    srcs = _two_year_sources(tmp_path)
    # training window: the 2024-12-30 run, and the 2025-01-02 run; the 2024-12-31 run arrives in 2025 (after the
    # 2024 source's range) and is dropped; the 2025-09-05 run is after the window
    e = delay_attrs.edge_delay_attributes(srcs, "2024-01-01", "2025-08-31", tt.node_of)
    assert len(e) == 1 and e.n_passages.iloc[0] == 2
    assert e.mean_arrival_delay.iloc[0] == pytest.approx((10.0 + 50.0) / 2)
    # a window that starts on the boundary sees only the 2025 source, never the spilled 2024 run
    e2 = delay_attrs.edge_delay_attributes(srcs, "2025-01-01", "2025-01-31", tt.node_of)
    assert e2.n_passages.iloc[0] == 1 and e2.mean_arrival_delay.iloc[0] == pytest.approx(50.0)
    # validation window sees nothing of the training data
    v = delay_attrs.edge_delay_attributes(srcs, "2025-09-01", "2025-10-31", tt.node_of)
    assert v.n_passages.iloc[0] == 1


def test_spill_across_the_validation_boundary_stays_out_of_the_training_window(tmp_path, tt):
    from src.splits import Source
    d = tmp_path / "s"
    rows = []
    # enters 2025-08-31 23:50, arrives at CSB 2025-09-01 00:10: validation, not training
    rows.append(dict(train="T", date="2025-08-31", seq=0, station="NDLS", arr=np.nan, dep=23 * 60 + 50, arr_delay=np.nan, dep_delay=1.0))
    rows.append(dict(train="T", date="2025-08-31", seq=1, station="CSB", arr=23 * 60 + 70, dep=23 * 60 + 71, arr_delay=99.0, dep_delay=99.0))
    _write_scenario(d, rows)
    srcs = [Source(str(d), "2025-01-01", "2025-12-31")]
    train = delay_attrs.edge_delay_attributes(srcs, "2024-01-01", "2025-08-31", tt.node_of)
    val = delay_attrs.edge_delay_attributes(srcs, "2025-09-01", "2025-10-31", tt.node_of)
    assert len(train) == 0 and val.n_passages.iloc[0] == 1 and val.mean_arrival_delay.iloc[0] == pytest.approx(99.0)


def test_weather_anchor_allows_runs_before_2025():
    from src.simulation.config import load_config
    from src.simulation.weather import SyntheticWeather
    w = SyntheticWeather(load_config(), 2024, anchor=-490)
    assert w.state(-366) in ("clear", "fog", "rain")
    assert SyntheticWeather(load_config(), 42).state(0) == SyntheticWeather(load_config(), 42, anchor=-120).state(0)
