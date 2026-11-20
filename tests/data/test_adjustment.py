"""Price adjustment with CRSP's distribution factors, and the CRSP return check."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from gnns.data import adjustment
from gnns.data import security_master as smod

NY = ZoneInfo("America/New_York")
CRSP_LAST = date(2025, 12, 31)
SM = pl.DataFrame(
    [("1", "permno", "10", date(2000, 1, 1), None, "crsp")], schema=smod.SCHEMA, orient="row"
)


def _days(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _dist(rows) -> pl.DataFrame:
    """(permno, ex-date, detail type, disfacshr)."""
    return pl.DataFrame(
        rows,
        schema={
            "permno": pl.Int64,
            "disexdt": pl.Date,
            "disdetailtype": pl.String,
            "disfacshr": pl.Float64,
        },
        orient="row",
    )


def _daily(rows) -> pl.DataFrame:
    """(permno, date, ret, price, previous price)."""
    return pl.DataFrame(
        rows,
        schema={
            "permno": pl.Int64,
            "dlycaldt": pl.Date,
            "dlyret": pl.Float64,
            "dlyprc": pl.Float64,
            "dlyprevprc": pl.Float64,
        },
        orient="row",
    )


def test_one_crsp_factor_per_distribution_day_covers_every_kind():
    d = _days(date(2020, 1, 1), 6)
    dist = _dist(
        [
            (10, d[1], "CASHD", 0.0),  # $1 cash dividend: 100 -> 99, return 0
            (10, d[2], "SPLIT", 1.0),  # 2-for-1: 99 -> 49.5, return 0
            (10, d[3], "SECSO", 0.0),  # spinoff: 49.5 -> 42, holder return +1%
            (10, d[4], "SECMRG", 0.0),  # ends the security: skipped
        ]
    )
    daily = _daily(
        [
            (10, d[0], 0.0, 100.0, 100.0),
            (10, d[1], 0.0, 99.0, 100.0),
            (10, d[2], 0.0, 49.5, 99.0),
            (10, d[3], 0.01, 42.0, 49.5),
            (10, d[4], 0.0, 42.0, 42.0),
            (10, d[5], 0.0, -21.0, 20.0),  # off 1 with no distribution: reported only
        ]
    )
    ev, unpriced, stray = adjustment.crsp_events(dist, daily, SM)
    assert ev.select("date", "factor", "split_factor").rows() == [
        (d[1], pytest.approx(0.99), pytest.approx(1.0)),
        (d[2], pytest.approx(0.5), pytest.approx(0.5)),
        (d[3], pytest.approx(42.0 / (49.5 * 1.01)), pytest.approx(1.0)),
    ]
    assert unpriced.is_empty()
    assert stray.rows() == [("1", d[5])]


def test_a_distribution_without_a_crsp_day_row_is_reported():
    d = date(2020, 1, 2)
    ev, unpriced, _ = adjustment.crsp_events(_dist([(10, d, "CASHD", 0.0)]), _daily([]), SM)
    assert ev.is_empty() and unpriced.rows() == [("1", d)]


def _raw(rows) -> pl.DataFrame:
    """(date, close, volume) of security 1, open = high = low = close."""
    return pl.DataFrame(
        [
            ("1", d, datetime(d.year, d.month, d.day, 16, tzinfo=NY), c, c, c, c, v)
            for d, c, v in rows
        ],
        schema={
            "security_id": pl.String,
            "date": pl.Date,
            "timestamp": pl.Datetime("us", "America/New_York"),
            **dict.fromkeys(adjustment.BAR_COLS, pl.Float64),
        },
        orient="row",
    )


def test_adjust_reproduces_total_returns_across_a_split_and_a_dividend():
    d = _days(date(2020, 1, 1), 4)
    # 2:1 split on d1 (100 -> 50), $1 dividend on d3 (ex-date close drops 51 -> 50).
    raw = _raw([(d[0], 100.0, 10.0), (d[1], 50.0, 20.0), (d[2], 51.0, 20.0), (d[3], 50.0, 20.0)])
    events = pl.DataFrame(
        {
            "security_id": ["1", "1"],
            "date": [d[1], d[3]],
            "factor": [0.5, 1 - 1 / 51],
            "split_factor": [0.5, 1.0],
        }
    )
    out = adjustment.adjust(raw, events)
    c = out["close"].to_list()
    assert c[3] == pytest.approx(50.0)  # the latest bar is unadjusted
    assert c[1] / c[0] - 1 == pytest.approx(0.0)  # the split is not a return
    assert c[3] / c[2] - 1 == pytest.approx(50 / 51 / (1 - 1 / 51) - 1)  # total return 0
    assert out["volume"].to_list() == pytest.approx([20.0, 20.0, 20.0, 20.0])
    assert out.columns == ["timestamp", "ticker", *adjustment.BAR_COLS]


def _prices(rows) -> pl.DataFrame:
    return pl.DataFrame(
        [(datetime(d.year, d.month, d.day, 16, tzinfo=NY), t, c) for t, d, c in rows],
        schema={
            "timestamp": pl.Datetime("us", "America/New_York"),
            "ticker": pl.String,
            "close": pl.Float64,
        },
        orient="row",
    )


def test_crsp_match_compares_member_sessions_only():
    d = _days(date(2024, 1, 2), 3)
    prices = _prices([("A", d[0], 100.0), ("A", d[1], 101.0), ("A", d[2], 90.0)])
    members = pl.DataFrame({"timestamp": prices["timestamp"][:2], "ticker": ["A", "A"]})
    sm = pl.DataFrame([("A", "permno", "5", d[0], None, "crsp")], schema=smod.SCHEMA, orient="row")
    daily = pl.DataFrame({"permno": [5, 5], "dlycaldt": [d[1], d[2]], "dlyret": [0.0101, 0.5]})
    shares, bad = adjustment.crsp_match(prices, members, sm, daily, CRSP_LAST)
    assert shares["compared"] == 1.0  # d2 is not a member session
    assert shares["within_0.001"] == 1.0
    assert bad.is_empty()


def test_build_needs_the_universe_build_first(tmp_path, monkeypatch):
    from gnns.data import wrds

    monkeypatch.setattr(
        wrds,
        "load",
        lambda root, name: (
            pl.DataFrame({c: [] for c in adjustment.EXTRACTS[name]}),
            {},
        ),
    )
    with pytest.raises(FileNotFoundError, match="run gnns build-universe first"):
        adjustment.build("u", date(2020, 1, 1), date(2020, 12, 31), tmp_path)
