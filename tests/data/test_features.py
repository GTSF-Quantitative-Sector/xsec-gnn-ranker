from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gnns.data import features

CFG = {
    "min_history_bars": 3,
    "list": [{"name": "momentum", "bars": 2}, {"name": "volatility", "bars": 3}],
}


def _prices(closes: dict[str, list[float | None]]) -> pl.DataFrame:
    """Daily bars from per-ticker close lists; None means no bar that day."""
    t0 = datetime(2024, 1, 1, 16)
    rows = [
        (t0 + timedelta(days=i), ticker, c)
        for ticker, cs in closes.items()
        for i, c in enumerate(cs)
        if c is not None
    ]
    return pl.DataFrame(rows, schema=["timestamp", "ticker", "close"], orient="row")


def test_values_match_hand_calculation():
    out = features.compute(_prices({"A": [10, 11, 12, 15, 9]}), CFG)
    assert out["timestamp"].dt.day().to_list() == [4, 5]  # first 3 bars are history
    assert out["momentum_2"].to_list() == pytest.approx([15 / 11 - 1, 9 / 12 - 1])
    rets = np.array([11 / 10, 12 / 11, 15 / 12, 9 / 15]) - 1
    expected = [np.std(rets[0:3], ddof=1), np.std(rets[1:4], ddof=1)]
    assert out["volatility_3"].to_list() == pytest.approx(expected)


def test_missing_bar_is_median_filled_across_tickers():
    closes = {"A": [10, 11, 12, 13], "B": [10, 10, 10, 11], "C": [10, 12, 12, None, 12]}
    out = features.compute(_prices(closes), {**CFG, "list": [{"name": "momentum", "bars": 1}]})
    day4 = out.filter(pl.col("timestamp").dt.day() == 4).sort("ticker")
    assert day4["ticker"].to_list() == ["A", "B", "C"]  # C keeps its node despite no bar
    assert day4["momentum_1"][2] == pytest.approx(np.median([13 / 12 - 1, 11 / 10 - 1]))


def test_late_listing_waits_for_its_own_history():
    out = features.compute(_prices({"A": [10] * 6, "B": [None, None, 10, 10, 10, 10]}), CFG)
    assert out.filter(pl.col("ticker") == "B")["timestamp"].dt.day().to_list() == [6]


def test_no_lookahead_truncating_the_future_changes_nothing():
    rng = np.random.default_rng(0)
    closes: dict[str, list[float | None]] = {
        t: [float(c) for c in 100 * np.cumprod(1 + rng.normal(0, 0.02, 40))] for t in "ABC"
    }
    closes["B"][20] = None
    prices = _prices(closes)
    cutoff = prices["timestamp"].sort()[25]
    full = features.compute(prices, CFG).filter(pl.col("timestamp") <= cutoff)
    truncated = features.compute(prices.filter(pl.col("timestamp") <= cutoff), CFG)
    assert_frame_equal(full, truncated)


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        ({**CFG, "list": [{"name": "momentun", "bars": 2}]}, "unknown features"),
        ({**CFG, "min_history_bars": 2}, "longest feature window"),
    ],
)
def test_bad_config_raises(cfg, message):
    with pytest.raises(ValueError, match=message):
        features.compute(_prices({"A": [10] * 5}), cfg)
