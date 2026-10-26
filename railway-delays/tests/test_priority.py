"""Priority sanity check in clear weather (SIMULATED). See the notebook section 6b and design doc section 12."""
from pathlib import Path

import pandas as pd
import pytest

if not (Path("data/processed/trains.parquet").exists() and Path("data/raw/trains.json").exists()):
    pytest.skip("Phase 2 outputs not present", allow_module_level=True)

from src.simulation import causes as C
from src.simulation.run import simulate, to_day
from src.simulation.weather import FixedWeather

ORDER = ["Rajdhani", "Duronto", "Shatabdi", "Superfast", "Mail/Express", "Passenger"]


@pytest.fixture(scope="module")
def clear_run():
    res = simulate(to_day("2025-06-01"), to_day("2025-06-14"), weather=FixedWeather("clear"))
    rec = res["recorder"]
    df, dfa = rec.tables(rec.months()[0])
    types = {t: s.type for t, s in res["specs"].items()}
    df["train"] = df.train.astype(str)
    df["type"] = df.train.map(types)
    d = df.dropna(subset=["arrival_delay"])
    raw = d.groupby("type").arrival_delay.mean().reindex(ORDER)
    ad = dfa[(dfa.phase == "arr") & dfa.cause_code.isin([C.SECTION_WAIT, C.YIELD, C.RAKE])].merge(df[["obs_id", "type"]], on="obs_id")
    adj = ad.groupby("type").new_minutes.sum().reindex(ORDER) / d.groupby("type").size().reindex(ORDER) * 100
    return raw, adj


@pytest.mark.xfail(strict=True, reason="Known: raw mean delay is not ordered by priority class because of exposure "
                                       "(Passenger and Mail/Express trains are short or start inside the corridor) and four "
                                       "Shatabdi trains stuck behind slower trains. See design doc section 12.")
def test_raw_mean_delay_is_ordered_by_priority(clear_run):
    raw, _ = clear_run
    assert max(raw.Rajdhani, raw.Duronto, raw.Shatabdi) <= raw.Superfast <= raw["Mail/Express"] <= raw.Passenger


def test_delay_caused_by_other_trains_per_exposure_follows_priority(clear_run):
    """Excluding the four-train Shatabdi class, delay added by other trains per 100 rows rises as priority falls."""
    _, adj = clear_run
    assert max(adj.Rajdhani, adj.Duronto) <= adj.Superfast <= adj["Mail/Express"] <= adj.Passenger


def test_the_priority_trains_are_no_longer_the_worst_when_overtaking_is_allowed(clear_run):
    raw, _ = clear_run
    assert raw.Rajdhani <= raw.Superfast and raw.Duronto <= raw.Superfast
