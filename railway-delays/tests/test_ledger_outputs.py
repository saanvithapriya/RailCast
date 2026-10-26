"""Cause tracking and output tables (all SIMULATED)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if not (Path("data/processed/trains.parquet").exists() and Path("data/raw/trains.json").exists()):
    pytest.skip("Phase 2 outputs not present", allow_module_level=True)

from src.simulation import causes as C
from src.simulation.config import load_config
from src.simulation.engine import MoveParams, Simulator
from src.simulation.ledger import Ledger
from src.simulation.run import simulate, to_day
from tests.helpers import FixedEntry, line_spec, make_net, run


@pytest.fixture(scope="module")
def short_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim")
    res = simulate(to_day("2025-01-12"), to_day("2025-01-14"), out_dir=out)
    return res, out


def test_ledger_components_sum_to_the_observed_delay(short_run):
    res, _ = short_run
    assert res["ledger"].checked > 10000
    assert res["ledger"].max_mismatch < 1e-6


def test_ledger_unit_additions_and_recovery():
    led = Ledger()
    net = make_net(3)
    s = Simulator(net, MoveParams(), ledger=led, delays=FixedEntry(20.0))
    r = run(s, line_spec("T", 3, 0, 100, last=2, entered=True))
    s.drain()
    # 20 min entry delay, recovers 6 then 6 more: 8 left at the last node, all of it entry delay
    assert led.max_mismatch < 1e-9


def test_attribution_sums_to_arrival_delay(short_run):
    res, _ = short_run
    rec = res["recorder"]
    df, dfa = rec.tables(rec.months()[0])
    arr = dfa[dfa.phase == "arr"]
    comp = arr[arr.cause_code > 0].groupby("obs_id").minutes.sum()
    late = df[(df.arrival_delay >= 0.5)].set_index("obs_id")
    both = late.join(comp, how="inner")
    assert len(both) > 1000
    # rows with attribution: components sum to the arrival delay (rows without attribution are inherited cluster rows)
    assert np.allclose(both.arrival_delay, both.minutes, atol=0.05)


def test_new_minutes_never_exceed_minutes_and_recovery_is_negative(short_run):
    res, _ = short_run
    rec = res["recorder"]
    _, dfa = rec.tables(rec.months()[0])
    pos = dfa[dfa.cause_code > 0]
    assert (pos.new_minutes <= pos.minutes + 1e-3).all()
    assert (dfa[dfa.cause_code == C.RECOVERY].minutes < 0).all()


def test_observation_columns_dtypes_and_labels(short_run):
    res, out = short_run
    files = list((out / "observations").glob("month=*/part.parquet"))
    assert files
    df = pd.read_parquet(files[0])
    for c in ["date", "train", "station", "seq", "scheduled_arrival", "scheduled_departure", "raw_scheduled_arrival",
              "raw_scheduled_departure", "actual_arrival", "actual_departure", "arrival_delay", "departure_delay",
              "primary_delay_injected", "cause_code", "cause_train", "root_event_id", "is_simulated"]:
        assert c in df.columns
    assert df.arrival_delay.dtype == np.float32 and str(df.train.dtype) == "category"
    assert df.is_simulated.all()
    import pyarrow.parquet as pq
    meta = pq.read_schema(files[0]).metadata
    assert meta[b"SIMULATED"] == b"true"
    for name in ("events.parquet", "weather.parquet", "config_used.yaml", "README_SIMULATED.txt"):
        assert (out / name).exists()
    assert "SIMULATED" in (out / "config_used.yaml").read_text()


def test_every_core_row_is_logged_once_per_run(short_run):
    res, _ = short_run
    rec = res["recorder"]
    df, _ = rec.tables(rec.months()[0])
    rows_per_train = {t: len(s.extra["rows"]) for t, s in res["specs"].items()}
    expected = sum(rows_per_train[t] for t in df.train.astype(str).unique()) * df.date.nunique()
    assert len(df) == pytest.approx(expected, rel=0.01)
    assert not df.duplicated(["train", "date", "seq"]).any()


def test_same_seed_gives_identical_output():
    a = simulate(to_day("2025-01-12"), to_day("2025-01-13"))
    b = simulate(to_day("2025-01-12"), to_day("2025-01-13"))
    da, _ = a["recorder"].tables(a["recorder"].months()[0])
    db, _ = b["recorder"].tables(b["recorder"].months()[0])
    pd.testing.assert_frame_equal(da, db)
    cfg = load_config(overrides={"simulation": {"seed": 7}})
    c = simulate(to_day("2025-01-12"), to_day("2025-01-13"), cfg)
    dc, _ = c["recorder"].tables(c["recorder"].months()[0])
    assert not np.allclose(da.arrival_delay.fillna(0), dc.arrival_delay.fillna(0))


def test_on_time_rows_have_zero_cause_and_late_rows_have_a_cause(short_run):
    res, _ = short_run
    rec = res["recorder"]
    df, _ = rec.tables(rec.months()[0])
    d = df.dropna(subset=["arrival_delay"])
    assert (d[d.arrival_delay < 0.5].cause_code == C.NONE).mean() > 0.9
    late = d[d.arrival_delay >= 0.5]
    assert (late.cause_code != C.NONE).mean() > 0.99
