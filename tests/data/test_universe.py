"""The universe build: symbols, the delisting cut, liquidity, market cap and the table."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from gnns.data import security_master as smod
from gnns.data import universe

NY = ZoneInfo("America/New_York")


def _bars(rows) -> pl.DataFrame:
    """(symbol, date, close[, volume[, vwap]]) -> unadjusted bars, open = high = low = close."""
    out = []
    for r in rows:
        sym, d, c = r[:3]
        v = r[3] if len(r) > 3 else 100.0
        vw = r[4] if len(r) > 4 else c
        out.append((sym, d, datetime(d.year, d.month, d.day, 16, tzinfo=NY), c, c, c, c, v, vw))
    return pl.DataFrame(
        out,
        schema={
            "symbol": pl.String,
            "date": pl.Date,
            "timestamp": pl.Datetime("us", "America/New_York"),
            **dict.fromkeys(universe.BAR_COLS, pl.Float64),
        },
        orient="row",
    )


def _days(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _between(a: date, b: date) -> list[date]:
    return _days(a, (b - a).days + 1)


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


# --- Symbols ------------------------------------------------------------------------------


def test_spellings_add_the_class_dot_both_ways():
    assert universe.spellings("BRKB", has_class=True) == ["BRKB", "BRK.B"]
    assert universe.spellings("BRK.B", has_class=True) == ["BRK.B", "BRKB"]
    assert universe.spellings("AAPL", has_class=False) == ["AAPL"]


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
    assert universe.choose_symbols(tw, bars)["symbol"].to_list() == ["BRK.B"]


def test_day_early_switch_keeps_the_old_symbol_when_the_new_one_is_another_security():
    # Datastream switches FB -> META on day 5; Massive's META belongs to an ETF until day 6.
    d = _days(date(2022, 6, 1), 10)
    iv = _intervals([("1", 0, "FB", d[0], d[4], "FB"), ("1", 1, "META", d[5], d[9], "META")])
    bars = _bars(
        [("FB", x, 190.0 + i) for i, x in enumerate(d[:6])]  # FB trades through day 5
        + [("META", x, 9.0) for x in d[:6]]  # the ETF
        + [("META", x, 184.0) for x in d[6:]]  # the real META from day 6
    )
    out = universe.assign_symbols(iv, bars, pl.Series(d))
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
    out = universe.assign_symbols(iv, bars, pl.Series(d))
    assert out["symbol"].to_list() == ["FOXA"] * 5 + ["TFCFA"] * 3  # no bar on d6, d7


def test_runs_compress_sessions_into_ticker_massive_rows():
    d = _days(date(2020, 1, 1), 4)
    assigned = pl.DataFrame({"security_id": ["1"] * 4, "date": d, "symbol": ["A", "A", "B", "B"]})
    rows = universe.runs(assigned)
    assert rows.select("value", "valid_from", "valid_to").rows() == [
        ("A", d[0], d[1]),
        ("B", d[2], d[3]),
    ]


# --- CRSP: delistings and market cap ------------------------------------------------------


def _sm(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=smod.SCHEMA, orient="row")


def _daily(rows) -> pl.DataFrame:
    """(permno, permco, date, price, cap, delisting flag)."""
    return pl.DataFrame(
        rows,
        schema={
            "permno": pl.Int64,
            "permco": pl.Int64,
            "dlycaldt": pl.Date,
            "dlyprc": pl.Float64,
            "dlycap": pl.Float64,
            "dlydelflg": pl.String,
        },
        orient="row",
    )


def test_crsp_ends_use_the_final_permno_and_its_last_trading_day():
    sm = _sm(
        [
            # A: its old PERMNO 1 was delisted (an exchange move), PERMNO 2 still trades.
            ("A", "permno", "1", date(2000, 1, 1), date(2019, 12, 31), "crsp"),
            ("A", "permno", "2", date(2020, 1, 1), None, "crsp"),
            ("B", "permno", "3", date(2000, 1, 1), date(2020, 2, 20), "crsp"),
            ("C", "permno", "4", date(2000, 1, 1), date(2020, 3, 2), "crsp"),
        ]
    )
    dl = pl.DataFrame(
        {
            "permno": [1, 3, 4],
            "delistingdt": [date(2019, 12, 31), date(2020, 2, 20), date(2020, 3, 2)],
            "delret": [0.0, -0.5, None],
        }
    )
    daily = _daily(
        [
            (3, 30, date(2020, 2, 19), 4.0, 10.0, "N"),
            (3, 30, date(2020, 2, 20), 2.0, 5.0, "Y"),  # the delisting record
            (4, 40, date(2020, 2, 28), 9.0, 10.0, "N"),
        ]
    )
    ends, no_return = universe.crsp_ends(sm, dl, daily, date(2025, 12, 31))
    assert ends.rows() == [("B", date(2020, 2, 19), -0.5), ("C", date(2020, 2, 28), None)]
    assert no_return["security_id"].to_list() == ["C"]


def test_bars_after_the_crsp_delisting_are_cut_and_reported():
    d = _days(date(2020, 2, 17), 6)
    raw = _bars([("XQ", x, 1.0) for x in d]).with_columns(security_id=pl.lit("B"))
    ends = pl.DataFrame({"security_id": ["B"], "last_day": [d[2]], "delret": [-0.5]})
    kept, dropped = universe.cut_at_delisting(raw, ends)
    assert kept["date"].to_list() == d[:3]
    assert dropped.select("security_id", "bars", "first", "last", "symbols").rows() == [
        ("B", 3, d[3], d[5], "XQ")
    ]


def test_market_cap_sums_every_class_of_the_company_in_dollars():
    day = date(2020, 3, 2)
    daily = _daily(
        [
            (1, 9, day, 10.0, 1_000.0, "N"),
            (2, 9, day, 10.0, 500.0, "N"),  # a second listed class
            (3, 9, day, 10.0, 99.0, "Y"),  # a delisting record is not a trading day
            (4, 7, day, 10.0, 200.0, "N"),
        ]
    )
    caps = universe.market_caps(daily).sort("permco")
    assert caps.rows() == [("7", day, 200_000.0), ("9", day, 1_500_000.0)]


def test_dollar_volume_is_vwap_times_volume_and_adv_trails_over_bars():
    d = _days(date(2020, 1, 1), 3)
    raw = _bars(
        [("A", d[0], 10.0, 100.0, 11.0), ("A", d[1], 10.0, 200.0, None), ("A", d[2], 10.0, 300.0)]
    ).with_columns(security_id=pl.lit("1"))
    liq = universe.liquidity(raw, (2,))
    assert liq["dollar_volume"].to_list() == [1100.0, 2000.0, 3000.0]  # close where no VWAP
    assert liq["adv_2"].to_list() == [1100.0, 1550.0, 2500.0]


# --- The table, end to end ----------------------------------------------------------------

J, F, M = date(2020, 1, 31), date(2020, 2, 28), date(2020, 3, 31)
OPEN = smod.DS_OPEN_END
START, END = J, M


def _world():
    """Four securities on three month-end lists: two classes of one company (1, 2), one
    delisted by CRSP on 2020-02-20 that kept trading on Massive (3), one joining in March
    (4). Bars and CRSP rows every calendar day."""
    spells = pl.DataFrame(
        [(4408.0, 1.0, J, M), (4408.0, 2.0, J, M), (4408.0, 3.0, J, F), (4408.0, 4.0, M, M)],
        schema=["indexlistintcode", "infocode", "startdate", "enddate"],
        orient="row",
    )
    far = date(2000, 1, 1)
    tickers = pl.DataFrame(
        [
            (1.0, far, OPEN, "AAA", "AAA"),
            (2.0, far, OPEN, "AAB", "AAB"),
            (3.0, far, date(2020, 2, 25), "CCC", "CCC"),
            (4.0, date(2020, 3, 1), OPEN, "DDD", "DDD"),
        ],
        schema=["infocode", "startdate", "enddate", "ticker", "dsmnem"],
        orient="row",
    )
    cusips = pl.DataFrame(
        [(float(i), far, OPEN, f"{i}" * 8 + "X") for i in (1, 2, 3, 4)],
        schema=["infocode", "startdate", "enddate", "cusip"],
        orient="row",
    )
    names = pl.DataFrame(
        [
            (1.0, far, OPEN, 9.0, "US1", "AAA CORP 'A'", None),
            (2.0, far, OPEN, 9.0, "US2", "AAA CORP 'B'", None),
            (3.0, far, OPEN, 33.0, "US3", "CCC INC", date(2020, 2, 25)),
            (4.0, far, OPEN, 44.0, "US4", "DDD INC", None),
        ],
        schema={
            "infocode": pl.Float64,
            "startdate": pl.Date,
            "enddate": pl.Date,
            "dscmpycode": pl.Float64,
            "isin": pl.String,
            "dsqtname": pl.String,
            "delistdate": pl.Date,
        },
        orient="row",
    )
    end_of_time = date(9999, 12, 31)
    security = pl.DataFrame(
        [
            (101, 900, far, end_of_time, "11111111", "AAA"),
            (102, 900, far, end_of_time, "22222222", "AAB"),
            (199, 900, far, end_of_time, "99999999", "AAC"),  # a class never on the list
            (103, 903, far, date(2020, 2, 20), "33333333", "CCC"),
            (104, 904, far, end_of_time, "44444444", "DDD"),
        ],
        schema=["permno", "permco", "secinfostartdt", "secinfoenddt", "cusip", "ticker"],
        orient="row",
    )
    delist = pl.DataFrame(
        {"permno": [103], "delistingdt": [date(2020, 2, 20)], "delret": [-0.5]},
        schema={"permno": pl.Int64, "delistingdt": pl.Date, "delret": pl.Float64},
    )
    days = _between(date(2020, 1, 21), M)
    daily_rows = [(p, 900, d, 10.0, 1_000.0, "N") for p in (101, 102, 199) for d in days]
    daily_rows += [(103, 903, d, 5.0, 500.0, "N") for d in days if d < date(2020, 2, 20)]
    daily_rows += [(103, 903, date(2020, 2, 20), 2.5, 250.0, "Y")]
    daily_rows += [(104, 904, d, 20.0, 2_000.0, "N") for d in days if d >= date(2020, 3, 1)]
    ex = {
        "ds_constituents": spells,
        "ds_tickers": tickers,
        "ds_cusips": cusips,
        "ds_names": names,
        "crsp_security": security,
        "crsp_delistings": delist,
        "crsp_daily": _daily(daily_rows),
    }
    bars = _bars(
        [("AAA", d, 10.0, 1_000.0) for d in days]
        + [("AAB", d, 10.0, 10.0) for d in days]
        + [("CCC", d, 5.0, 100.0) for d in days if d <= date(2020, 2, 25)]
        + [("DDD", d, 20.0, 100.0) for d in days if d >= date(2020, 3, 1)]
    )
    return ex, bars


SPEC = {"list_code": 4408, "adv_windows": [2]}


def test_the_table_has_every_trading_session_with_membership_ids_and_metrics(tmp_path):
    ex, bars = _world()
    b = universe.assemble("u", SPEC, START, END, ex, bars, tmp_path)
    t = b.universe
    assert t.columns == list(universe.schema((2,)))
    on = {
        d: sorted(g["security_id"].to_list())
        for (d,), g in t.filter("member").group_by(pl.col("timestamp").dt.date())
    }
    assert on[J] == ["1", "2", "3"]  # both share classes are members
    assert on[date(2020, 2, 19)] == ["1", "2", "3"]
    assert on[date(2020, 2, 20)] == ["1", "2"]  # 3's bars end at its last CRSP trading day
    assert on[M] == ["1", "2", "4"]  # 4 joins on the March list
    row = t.filter((pl.col("security_id") == "1") & (pl.col("timestamp").dt.date() == F)).row(
        0, named=True
    )
    assert row["list_date"] == F and row["ticker_massive"] == "AAA" and row["company"] == "9"
    assert (row["permno"], row["permco"], row["cusip"]) == ("101", "900", "11111111")
    assert row["dollar_volume"] == pytest.approx(10_000.0) and row["adv_2"] == pytest.approx(
        10_000.0
    )
    assert row["mktcap"] == pytest.approx(3_000_000.0)  # 101 + 102 + 199, in dollars
    assert t.filter(pl.col("security_id") == "2")["mktcap"].unique().to_list() == [3_000_000.0]
    # 4 trades from March 1 but joins on the March 31 list: its earlier rows carry its
    # identifiers too (for lookbacks), as non-members.
    four = t.filter(pl.col("security_id") == "4").sort("timestamp")
    assert four["timestamp"].dt.date().min() == date(2020, 3, 1)
    assert four["member"].to_list() == [False] * 30 + [True]
    assert four["ticker_massive"].unique().to_list() == ["DDD"]
    feb = t.filter(pl.col("timestamp").dt.date().is_between(date(2020, 2, 1), date(2020, 2, 27)))
    assert feb["list_date"].unique().to_list() == [J]

    assert b.delistings.select("ticker", "delret").rows() == [("3", -0.5)]
    assert b.delistings["last_bar"].dt.date().to_list() == [date(2020, 2, 19)]
    assert b.reports["bars_after_crsp_delisting"].select("security_id", "bars").rows() == [("3", 6)]
    last3 = b.raw_bars.filter(pl.col("security_id") == "3")["timestamp"].dt.date().max()
    assert last3 == date(2020, 2, 19)
    assert sorted(b.raw_bars["security_id"].unique()) == ["1", "2", "3", "4"]
    assert b.raw_bars.columns == ["timestamp", "security_id", *universe.BAR_COLS]
    assert b.reports["member_sessions_without_symbol"].is_empty()


def test_the_table_at_t_uses_nothing_after_t(tmp_path):
    """Look-ahead: cutting every input after d changes no row at t <= d."""
    d = date(2020, 2, 15)
    ex, bars = _world()
    full = universe.assemble("u", SPEC, START, END, ex, bars, tmp_path).universe
    cut = dict(ex)
    cut["ds_constituents"] = (
        ex["ds_constituents"]
        .filter(pl.col("startdate") <= d)
        .with_columns(enddate=pl.min_horizontal("enddate", pl.lit(J)))
    )
    cut["crsp_daily"] = ex["crsp_daily"].filter(pl.col("dlycaldt") <= d)
    cut["crsp_delistings"] = ex["crsp_delistings"].filter(pl.col("delistingdt") <= d)
    part = universe.assemble(
        "u", SPEC, START, d, cut, bars.filter(pl.col("date") <= d), tmp_path
    ).universe
    upto = full.filter(pl.col("timestamp").dt.date() <= d)
    assert part.equals(upto)


def test_a_cache_without_the_new_columns_says_to_refetch(tmp_path, monkeypatch):
    from gnns.data import wrds

    monkeypatch.setattr(wrds, "load", lambda root, name: (pl.DataFrame({"permno": [1]}), {}))
    with pytest.raises(ValueError, match="re-run gnns fetch-wrds"):
        universe.load_extracts(tmp_path)


def test_build_refuses_an_end_after_crsp(tmp_path, monkeypatch):
    ex, _ = _world()
    monkeypatch.setattr(universe, "load_extracts", lambda root: ex)
    with pytest.raises(ValueError, match="after CRSP's last date 2020-03-31"):
        universe.build("u", SPEC, START, date(2020, 4, 30), tmp_path, tmp_path)
