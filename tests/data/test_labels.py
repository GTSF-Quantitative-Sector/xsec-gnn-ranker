from datetime import datetime, timedelta

import polars as pl
import pytest

from gnns.data.labels import forward_returns

T0 = datetime(2024, 1, 1, 16)


def _prices(opens: dict[str, list[float | None]]) -> pl.DataFrame:
    """Bars from per-ticker open lists (close = open * 2, to catch close misuse)."""
    rows = [
        (T0 + timedelta(days=i), ticker, o, o * 2)
        for ticker, os in opens.items()
        for i, o in enumerate(os)
        if o is not None
    ]
    return pl.DataFrame(rows, schema=["timestamp", "ticker", "open", "close"], orient="row")


def test_next_open_to_open_h_bars_later():
    out = forward_returns(_prices({"A": [10, 11, 12, 15, 9, 8]}), horizon_bars=2)
    # t=day0: open[1]=11 -> open[3]=15; t=day1: 12 -> 9; t=day2: 15 -> 8; later t have no exit.
    assert out["fwd_return"].to_list() == pytest.approx([15 / 11 - 1, 9 / 12 - 1, 8 / 15 - 1])
    assert out["label_end"].to_list() == [T0 + timedelta(days=d) for d in (3, 4, 5)]


def test_missing_entry_or_exit_bar_gives_no_label():
    out = forward_returns(_prices({"A": [10, 11, 12, 13, 14], "B": [10, None, 12, 13, 14]}), 1)
    bars = (pl.col("timestamp") - T0).dt.total_days()
    b = out.filter(pl.col("ticker") == "B").select(bars)["timestamp"].to_list()
    # B has no bar 1: bar 0 loses its entry. Bar 1 itself still gets a label (entry is bar 2).
    assert b == [1, 2]
    assert out.filter(pl.col("ticker") == "A").height == 3


def test_label_ignores_bar_t_itself():
    base: dict[str, list[float | None]] = {"A": [10.0, 11.0, 12.0, 13.0]}
    changed: dict[str, list[float | None]] = {"A": [99.0, 11.0, 12.0, 13.0]}  # only bar 0
    a, b = forward_returns(_prices(base), 1), forward_returns(_prices(changed), 1)
    assert a.filter(pl.col("timestamp") == T0)["fwd_return"].item() == pytest.approx(
        b.filter(pl.col("timestamp") == T0)["fwd_return"].item()
    )


def _delist(ticker: str, day: int, delret: float) -> pl.DataFrame:
    return pl.DataFrame(
        {"ticker": [ticker], "last_bar": [T0 + timedelta(days=day)], "delret": [delret]}
    )


def test_a_delisting_inside_the_window_labels_from_the_final_value():
    # #3's repro: X trades days 0-3 of a 10-day axis; horizon 5 -> day 0 exits on day 6.
    prices = _prices({"A": [10.0 + i for i in range(10)], "X": [4.0, 4.0, 4.0, 5.0]})
    plain = forward_returns(prices, 5)
    assert plain.filter(pl.col("ticker") == "X").is_empty()
    out = forward_returns(prices, 5, _delist("X", 3, -0.5)).filter(pl.col("ticker") == "X")
    bars = (pl.col("timestamp") - T0).dt.total_days()
    # Final value = close[3] * (1 + delret) = 10 * 0.5 = 5. Days 0-2 enter at open[t+1] = 4
    # (day 2 enters at day 3's open, 5); day 3 is the last bar, so its label is delret.
    assert out.select(bars)["timestamp"].to_list() == [0, 1, 2, 3]
    assert out["fwd_return"].to_list() == pytest.approx([5 / 4 - 1, 5 / 4 - 1, 5 / 5 - 1, -0.5])
    assert out["label_end"].to_list() == [T0 + timedelta(days=d) for d in (6, 7, 8, 9)]


def test_delisting_labels_respect_the_trailing_cutoff_and_leave_others_alone():
    prices = _prices({"A": [10.0 + i for i in range(10)], "X": [4.0] * 8})
    out = forward_returns(prices, 5, _delist("X", 7, 0.0))
    # Positions with i + 1 + 5 > 9 (days 4..9) have no label, delisted or not.
    assert out["timestamp"].max() == T0 + timedelta(days=3)
    a_plain = forward_returns(prices, 5).filter(pl.col("ticker") == "A")
    assert out.filter(pl.col("ticker") == "A").equals(a_plain)


def test_a_missing_bar_without_a_delisting_stays_missing():
    prices = _prices({"A": [10.0 + i for i in range(10)], "X": [4.0, 4.0, 4.0, 5.0]})
    out = forward_returns(prices, 5, _delist("A", 9, 0.0))
    assert out.filter(pl.col("ticker") == "X").is_empty()
