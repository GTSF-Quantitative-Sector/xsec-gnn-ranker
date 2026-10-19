from datetime import date, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gnns.data import store


def _prices(rows: list[tuple[date, str, float]]) -> pl.DataFrame:
    """Flat bars (open = high = low = close) from (date, ticker, price) tuples."""
    return pl.DataFrame(
        {
            "timestamp": [datetime(d.year, d.month, d.day, 16) for d, _, _ in rows],
            "ticker": [t for _, t, _ in rows],
            **{c: [p for _, _, p in rows] for c in store.OHLC},
            "volume": [100.0] * len(rows),
        }
    ).with_columns(pl.col("timestamp").dt.replace_time_zone("America/New_York"))


def test_adjust_uses_first_ex_date_after_bar():
    prices = _prices([(date(2024, 1, d), t, 10.0) for d in (2, 3, 4, 8) for t in ("A", "B")])
    divs = pl.DataFrame(
        {
            "ticker": ["A", "A"],
            "ex_date": [date(2024, 1, 3), date(2024, 1, 5)],
            "factor": [0.5, 0.8],
        }
    )
    out = store.adjust(prices, divs)
    assert out.columns == store.PRICE_COLS
    # The Jan 3 bar is on an ex-date, so it takes the next one's factor (Jan 5).
    assert out.filter(pl.col("ticker") == "A")["close"].to_list() == [5.0, 8.0, 8.0, 10.0]
    assert out.filter(pl.col("ticker") == "B")["close"].to_list() == [10.0] * 4


def test_validate_passes_clean_frame():
    clean = _prices([(date(2024, 1, 2), "A", 10.0)])
    store.validate(clean, ["A"], date(2024, 1, 1), date(2024, 1, 31))


def test_validate_reports_every_violation():
    bad = pl.concat(
        [
            _prices([(date(2024, 1, 2), "A", 10.0)] * 2),  # duplicate
            _prices([(date(2024, 1, 3), "A", 10.0)]).with_columns(high=pl.lit(9.0)),
            _prices([(date(2025, 1, 2), "A", 10.0)]),  # outside window
        ]
    )
    with pytest.raises(ValueError) as exc:
        store.validate(bad, ["A", "B"], date(2024, 1, 1), date(2024, 12, 31))
    for expected in ("duplicate", "bad OHLC", "outside", "no bars: ['B']"):
        assert expected in str(exc.value)


def test_write_then_load_round_trips(tmp_path):
    prices = _prices([(date(2024, 1, d), "A", 10.0) for d in (2, 3)])
    store.write(prices, tmp_path / "u" / "1d", {"universe": "u"})
    assert_frame_equal(store.load(tmp_path, "u", "1d").collect(), prices)
    assert store.load(tmp_path, "u", "1d", start=date(2024, 1, 3)).collect().height == 1
    assert not list((tmp_path / "u" / "1d").glob("*.tmp"))
