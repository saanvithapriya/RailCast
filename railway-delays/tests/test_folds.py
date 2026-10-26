"""Rolling-origin folds: ranges from the config, and no leakage from a fold's validation data into its training data."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if not (Path("data/processed/trains.parquet").exists() and Path("data/raw/trains.json").exists()):
    pytest.skip("Phase 2 outputs not present", allow_module_level=True)

from src import splits
from src.graph import delay_attrs
from src.graph.timetable import load_timetable
from tests.test_graph import _write_scenario


@pytest.fixture(scope="module")
def tt():
    return load_timetable()


def test_fold_ranges_come_from_the_config():
    f = splits.folds()
    assert splits.fold_names() == ["fold1_fog", "fold2_clear"]
    assert f["fold1_fog"] == {"train": ("2024-01-01", "2024-10-31"), "validate": ("2024-11-01", "2024-12-31")}
    assert f["fold2_clear"] == {"train": ("2024-01-01", "2025-08-31"), "validate": ("2025-09-01", "2025-10-31")}
    assert splits.eda_windows() == {"decision": ("2024-01-01", "2024-10-31"), "descriptive": ("2024-01-01", "2025-08-31")}


def test_folds_are_ordered_in_time_and_leave_the_test_window_alone():
    d = pd.Timestamp
    test_start = d(splits.window("test")[0])
    for name, f in splits.folds().items():
        assert d(f["train"][1]) + pd.Timedelta(days=1) == d(f["validate"][0]), name      # validation starts right after training
        assert d(f["validate"][1]) < test_start                                          # test window never touched
        assert d(f["train"][0]) >= d(splits.window("train")[0]) and d(f["train"][1]) <= d(splits.window("train")[1])
    assert splits.fold("fold2_clear")["train"] == splits.window("train")                  # fold 2 trains on the full training window
    assert splits.eda_windows()["decision"] == splits.fold("fold1_fog")["train"]


def test_fold_validation_fog_coverage_matches_the_documented_counts():
    if not all(Path(s.dir, "weather.parquet").exists() for s in splits.sources("medium")):
        pytest.skip("weather tables not present")
    w = pd.concat([pd.read_parquet(Path(s.dir) / "weather.parquet") for s in splits.sources("medium")])
    w["date"] = pd.to_datetime(w.date)
    fog = lambda a, b: int(((w.date >= a) & (w.date <= b) & (w.state == "fog")).sum())
    assert fog(*splits.fold("fold1_fog")["validate"]) == 12
    assert fog(*splits.fold("fold2_clear")["validate"]) == 0


def _fold1_sources(tmp_path):
    d = tmp_path / "s"
    rows = []
    # train range ends 2024-10-31. Run A is entirely inside; run B enters 2024-10-31 23:50 and arrives at CSB at
    # 2024-11-01 00:10 (midnight spill into the validation range); run C is a validation run.
    for date, base, delay in (("2024-10-30", 600, 5.0), ("2024-10-31", 23 * 60 + 50, 77.0), ("2024-11-05", 600, 40.0)):
        rows.append(dict(train="T", date=date, seq=0, station="NDLS", arr=np.nan, dep=base, arr_delay=np.nan, dep_delay=1.0))
        rows.append(dict(train="T", date=date, seq=1, station="CSB", arr=base + 20, dep=base + 21, arr_delay=delay, dep_delay=delay))
    _write_scenario(d, rows)
    return [splits.Source(str(d), "2024-01-01", "2024-12-31")]


def test_validation_data_never_enters_the_training_tables_including_midnight_spill(tmp_path, tt):
    srcs = _fold1_sources(tmp_path)
    nd, ed = delay_attrs.fold_train_delay_tables("fold1_fog", tt.node_of, sources=srcs)
    assert len(ed) == 1 and ed.n_passages.iloc[0] == 1                        # only run A
    assert ed.mean_arrival_delay.iloc[0] == pytest.approx(5.0)               # the spilled run (77) and the validation run (40) are out
    # the validation tables get exactly the spilled run and the validation run
    nv, ev = delay_attrs.fold_train_delay_tables("fold1_fog", tt.node_of, part="validate", sources=srcs)
    assert ev.n_passages.iloc[0] == 2 and ev.mean_arrival_delay.iloc[0] == pytest.approx((77.0 + 40.0) / 2)
    csb = nd[nd.node == tt.node_of["CSB"]].iloc[0]
    assert csb.n_arrivals == 1 and csb.mean_arrival_delay == pytest.approx(5.0)


def test_changing_validation_data_cannot_change_the_training_tables(tmp_path, tt):
    srcs = _fold1_sources(tmp_path)
    nd1, ed1 = delay_attrs.fold_train_delay_tables("fold1_fog", tt.node_of, sources=srcs)
    p = Path(srcs[0].dir) / "observations" / "month=2024-11" / "part.parquet"
    df = pd.read_parquet(p)
    df["arrival_delay"] = np.float32(999)
    df["departure_delay"] = np.float32(999)
    df.to_parquet(p, index=False)
    nd2, ed2 = delay_attrs.fold_train_delay_tables("fold1_fog", tt.node_of, sources=srcs)
    pd.testing.assert_frame_equal(ed1, ed2)
    pd.testing.assert_frame_equal(nd1, nd2)


def test_fold2_training_tables_are_the_full_training_window(tmp_path, tt):
    srcs = _fold1_sources(tmp_path)
    nd, ed = delay_attrs.fold_train_delay_tables("fold2_clear", tt.node_of, sources=srcs)
    assert ed.n_passages.iloc[0] == 3                                         # nothing here is after 2025-08-31


def test_real_fold1_training_tables_do_not_depend_on_anything_after_2024_10_31(tt):
    """Same fold-1 training tables whether or not the data after 2024-10-31 exists (the source is clipped at the boundary)."""
    if not Path(splits.sources("medium")[0].dir, "observations").exists():
        pytest.skip("2024 run not present")
    full = splits.sources("medium")
    a, b = splits.fold("fold1_fog")["train"]
    clipped = [splits.Source(full[0].dir, full[0].start, b, full[0].seed)]          # nothing after the boundary
    nd1, ed1 = delay_attrs.fold_train_delay_tables("fold1_fog", tt.node_of, sources=full)
    nd2, ed2 = delay_attrs.fold_train_delay_tables("fold1_fog", tt.node_of, sources=clipped)
    pd.testing.assert_frame_equal(ed1, ed2)
    pd.testing.assert_frame_equal(nd1, nd2)
    assert ed1.n_passages.sum() > 3_000_000
