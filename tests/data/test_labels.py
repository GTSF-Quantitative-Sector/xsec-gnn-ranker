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
