"""The sp500 build's pure steps: symbols, event factors, adjustment, delistings, checks."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from gnns.data import equities
from gnns.data import security_master as smod

NY = ZoneInfo("America/New_York")
CRSP_LAST = date(2025, 12, 31)


def _bars(rows) -> pl.DataFrame:
    """(symbol, date, close[, volume]) -> unadjusted bars with open = high = low = close."""
    out = []
    for r in rows:
        sym, d, c = r[:3]
        v = r[3] if len(r) > 3 else 100.0
        out.append((sym, d, datetime(d.year, d.month, d.day, 16, tzinfo=NY), c, c, c, c, v))
    return pl.DataFrame(
        out,
        schema={
            "symbol": pl.String,
            "date": pl.Date,
            "timestamp": pl.Datetime("us", "America/New_York"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
        orient="row",
    )


def _days(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _intervals(rows) -> pl.DataFrame:
    """(security_id, k, ticker, vf, vt, symbol)."""
    return pl.DataFrame(
        rows,
        schema={
            "security_id": pl.String,
            "k": pl.Int64,
            "ticker": pl.String,
            "vf": pl.Date,
            "vt": pl.Date,
            "symbol": pl.String,
        },
        orient="row",
    )


def test_spellings_add_the_class_dot_both_ways():
    assert equities.spellings("BRKB", has_class=True) == ["BRKB", "BRK.B"]
    assert equities.spellings("BRK.B", has_class=True) == ["BRK.B", "BRKB"]
    assert equities.spellings("AAPL", has_class=False) == ["AAPL"]


def test_choose_symbols_takes_the_spelling_with_bars():
    tw = pl.DataFrame(
        {
            "security_id": ["1"],
            "k": [0],
            "ticker": ["BRKB"],
            "vf": [date(2020, 1, 1)],
            "vt": [date(2020, 1, 10)],
            "candidates": [["BRKB", "BRK.B"]],
        }
    )
    bars = _bars([("BRK.B", d, 200.0) for d in _days(date(2020, 1, 1), 10)])
    assert equities.choose_symbols(tw, bars)["symbol"].to_list() == ["BRK.B"]


def test_day_early_switch_keeps_the_old_symbol_when_the_new_one_is_another_security():
    # Datastream switches FB -> META on day 5; Massive's META belongs to an ETF until day 6.
    d = _days(date(2022, 6, 1), 10)
    iv = _intervals([("1", 0, "FB", d[0], d[4], "FB"), ("1", 1, "META", d[5], d[9], "META")])
    bars = _bars(
        [("FB", x, 190.0 + i) for i, x in enumerate(d[:6])]  # FB trades through day 5
        + [("META", x, 9.0) for x in d[:6]]  # the ETF
        + [("META", x, 184.0) for x in d[6:]]  # the real META from day 6
    )
    out = equities.assign_symbols(iv, bars, pl.Series(d))
    assert out["symbol"].to_list() == ["FB"] * 6 + ["META"] * 4


def test_a_handover_never_borrows_a_discontinuous_symbol():
    # The old security (TFCFA) trades one last day; FOXA now belongs to a new company.
    d = _days(date(2019, 3, 14), 8)
    iv = _intervals([("1", 0, "FOXA", d[0], d[4], "FOXA"), ("1", 1, "TFCFA", d[5], d[7], "TFCFA")])
    bars = _bars(
        [("FOXA", x, 51.0) for x in d[:5]]
        + [("TFCFA", d[5], 49.7)]
        + [("FOXA", x, 39.0) for x in d[5:]]  # the new company's FOXA
    )
    out = equities.assign_symbols(iv, bars, pl.Series(d))
    assert out["symbol"].to_list() == ["FOXA"] * 5 + ["TFCFA"] * 3  # no bar on d6, d7


def test_runs_compress_sessions_into_ticker_massive_rows():
    d = _days(date(2020, 1, 1), 4)
    assigned = pl.DataFrame({"security_id": ["1"] * 4, "date": d, "symbol": ["A", "A", "B", "B"]})
    rows = equities.runs(assigned)
    assert rows.select("value", "valid_from", "valid_to").rows() == [
        ("A", d[0], d[1]),
        ("B", d[2], d[3]),
    ]


def _used(symbol: str, days: list[date], sid: str = "1") -> pl.DataFrame:
    return pl.DataFrame(
        {"security_id": [sid] * len(days), "date": days, "symbol": [symbol] * len(days)}
    )


def test_split_events_and_odd_ratios():
    d = _days(date(2015, 7, 1), 30)
    splits = pl.DataFrame(
        {
            "symbol": ["EBAY", "EBAY", "XYZ"],
            "date": [date(2015, 7, 20), date(2015, 7, 10), date(2015, 7, 15)],
            "split_from": [1000.0, 1.0, 1.0],
            "split_to": [2376.0, 2.0, 2.0],
        }
    )
    ev, odd = equities.split_events(splits, _used("EBAY", d))
    assert sorted(ev["factor"].to_list()) == pytest.approx([1000 / 2376, 0.5])
    assert odd["date"].to_list() == [date(2015, 7, 20)]  # 1000:2376 is not a round ratio


def test_dividend_factor_uses_the_unadjusted_close_before_the_ex_date():
    d = _days(date(2020, 1, 1), 5)
    raw = _bars([("A", x, 100.0) for x in d]).rename({"symbol": "security_id"})
    raw = raw.with_columns(security_id=pl.lit("1"))
    divs = pl.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "date": [d[3], d[0], d[2]],
            "cash_amount": [2.0, 1.0, 1.0],
            "currency": ["USD", "USD", "CAD"],
            "distribution_type": ["recurring"] * 3,
        }
    )
    ev, skipped = equities.dividend_events(divs, _used("A", d), raw)
    assert ev.select("date", "factor").rows() == [(d[3], pytest.approx(0.98))]
    assert sorted(skipped["date"].to_list()) == [d[0], d[2]]  # no earlier bar; not USD


def test_spinoff_factor_is_crsp_whole_day_and_terminal_distributions_are_ignored():
    dist = pl.DataFrame(
        {
            "permno": [10, 10],
            "disexdt": [date(2015, 7, 20), date(2016, 1, 4)],
            "distype": ["SP", "SP"],
            "disdetailtype": ["SECSO", "SECMRG"],
            "disfacpr": [1.4, -1.0],
            "disfacshr": [0.0, 0.0],
            "disdivamt": [0.0, 0.0],
        }
    )
    daily = pl.DataFrame(
        {
            "permno": [10],
            "dlycaldt": [date(2015, 7, 20)],
            "dlyret": [0.01],  # total return including the spun-off shares
            "dlyprc": [28.0],
            "dlyprevprc": [65.0],
        }
    )
    sm = pl.DataFrame(
        [("1", "permno", "10", date(2000, 1, 1), None, "crsp")], schema=smod.SCHEMA, orient="row"
    )
    ev, unpriced = equities.spinoff_events(dist, daily, sm, CRSP_LAST)
    assert ev.select("security_id", "date").rows() == [("1", date(2015, 7, 20))]
    assert ev["factor"].item() == pytest.approx(28.0 / (65.0 * 1.01))
    assert unpriced.is_empty()


def test_combine_drops_massive_events_on_a_spinoff_day():
    day = date(2015, 7, 20)
    split = pl.DataFrame({"security_id": ["1"], "date": [day], "factor": [0.42], "kind": ["split"]})
    div = pl.DataFrame(
        {"security_id": ["1"], "date": [day], "factor": [0.99], "kind": ["dividend"]}
    )
    spin = pl.DataFrame(
        {"security_id": ["1"], "date": [day], "factor": [0.43], "kind": ["spinoff"]}
    )
    other = pl.DataFrame(
        {"security_id": ["1"], "date": [date(2016, 1, 4)], "factor": [0.5], "kind": ["split"]}
    )
    ev = equities.combine(split, div, spin, other)
    assert ev.select("date", "factor", "split_factor").rows() == [
        (day, pytest.approx(0.43), pytest.approx(1.0)),
        (date(2016, 1, 4), pytest.approx(0.5), pytest.approx(0.5)),
    ]


def test_adjust_reproduces_total_returns_across_a_split_and_a_dividend():
    d = _days(date(2020, 1, 1), 4)
    # 2:1 split on d1 (100 -> 50), $1 dividend on d3 (ex-date close drops 51 -> 50).
    raw = _bars(
        [
            ("A", d[0], 100.0, 10.0),
            ("A", d[1], 50.0, 20.0),
            ("A", d[2], 51.0, 20.0),
            ("A", d[3], 50.0, 20.0),
        ]
    )
    raw = raw.with_columns(security_id=pl.lit("1"))
    events = equities.combine(
        pl.DataFrame({"security_id": ["1"], "date": [d[1]], "factor": [0.5], "kind": ["split"]}),
        pl.DataFrame(
            {"security_id": ["1"], "date": [d[3]], "factor": [1 - 1 / 51], "kind": ["dividend"]}
        ),
    )
    out = equities.adjust(raw, events)
    c = out["close"].to_list()
    assert c[3] == pytest.approx(50.0)  # the latest bar is unadjusted
    assert c[1] / c[0] - 1 == pytest.approx(0.0)  # the split is not a return
    assert c[3] / c[2] - 1 == pytest.approx(50 / 51 / (1 - 1 / 51) - 1)  # total return 0
    assert out["volume"].to_list() == pytest.approx([20.0, 20.0, 20.0, 20.0])


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


def test_delistings_use_crsp_then_the_last_trade():
    prices = _prices(
        [
            ("A", date(2019, 3, 19), 49.0),  # CRSP delisting
            ("B", date(2023, 9, 29), 0.1),  # traded OTC past its CRSP record
            ("C", date(2026, 2, 3), 20.0),  # after CRSP's last date
            ("D", date(2026, 6, 30), 10.0),  # still trading
        ]
    )
    sm = pl.DataFrame(
        [
            ("A", "permno", "1", date(2000, 1, 1), date(2019, 3, 20), "crsp"),
            ("B", "permno", "2", date(2000, 1, 1), date(2023, 5, 1), "crsp"),
        ],
        schema=smod.SCHEMA,
        orient="row",
    )
    crsp = pl.DataFrame(
        {
            "permno": [1, 2],
            "delistingdt": [date(2019, 3, 20), date(2023, 5, 1)],
            "delret": [0.02, -0.9],
        }
    )
    out, review = equities.delistings(prices, sm, crsp, CRSP_LAST, date(2026, 6, 30))
    assert out.select("ticker", "delret", "source", "provisional").rows() == [
        ("A", pytest.approx(0.02), "crsp", False),
        ("B", 0.0, "last_trade", False),
        ("C", 0.0, "last_trade", True),
    ]
    assert review["ticker"].to_list() == ["B"]


def test_crsp_match_compares_member_sessions_only():
    d = _days(date(2024, 1, 2), 3)
    prices = _prices([("A", d[0], 100.0), ("A", d[1], 101.0), ("A", d[2], 90.0)])
    members = pl.DataFrame({"timestamp": prices["timestamp"][:2], "ticker": ["A", "A"]})
    sm = pl.DataFrame([("A", "permno", "5", d[0], None, "crsp")], schema=smod.SCHEMA, orient="row")
    daily = pl.DataFrame({"permno": [5, 5], "dlycaldt": [d[1], d[2]], "dlyret": [0.0101, 0.5]})
    shares, bad = equities.crsp_match(prices, members, sm, daily, CRSP_LAST)
    assert shares["compared"] == 1.0  # d2 is not a member session
    assert shares["within_0.001"] == 1.0
    assert bad.is_empty()
