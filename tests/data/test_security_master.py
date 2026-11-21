"""Security master: derivation rules, overrides, invariants and lookups (fixtures only)."""

from datetime import date

import polars as pl
import pytest

from gnns.data import security_master as smod

OPEN = smod.DS_OPEN_END
CRSP_LAST = date(2025, 12, 31)


DATED = {"infocode": pl.Float64, "startdate": pl.Date, "enddate": pl.Date}


def _names(rows):
    schema = DATED | {
        "dscmpycode": pl.Float64,
        "isin": pl.String,
        "dsqtname": pl.String,
        "delistdate": pl.Date,
    }
    return pl.DataFrame(rows, schema=schema, orient="row")


def _dated(rows, value: str):
    return pl.DataFrame(rows, schema=DATED | {value: pl.String}, orient="row")


def _crsp(rows):
    schema = {
        "permno": pl.Int64,
        "permco": pl.Int64,
        "secinfostartdt": pl.Date,
        "secinfoenddt": pl.Date,
        "cusip": pl.String,
        "ticker": pl.String,
    }
    return pl.DataFrame(rows, schema=schema, orient="row")


def _derive(names=None, tickers=None, cusips=None, crsp=None):
    return smod.derive(
        names if names is not None else _names([]),
        tickers if tickers is not None else _dated([], "ticker"),
        cusips if cusips is not None else _dated([], "cusip"),
        crsp if crsp is not None else _crsp([]),
        CRSP_LAST,
    )


def _of(sm, sid, id_type):
    return sm.filter((pl.col("security_id") == sid) & (pl.col("id_type") == id_type)).sort(
        "valid_from"
    )


def test_datastream_rows_keep_their_dates_and_open_end_means_null():
    sm = _derive(
        names=_names([(1.0, date(2007, 3, 1), OPEN, 77.0, "US0001", "ACME", None)]),
        tickers=_dated(
            [
                (1.0, date(2000, 1, 3), date(2018, 2, 26), "OLD"),
                (1.0, date(2018, 2, 27), OPEN, "NEW"),
            ],
            "ticker",
        ),
    )
    t = _of(sm, "1", "ticker_datastream")
    assert t["value"].to_list() == ["OLD", "NEW"]
    assert t["valid_to"].to_list() == [date(2018, 2, 26), None]
    assert _of(sm, "1", "company")["value"].to_list() == ["77"]


def test_ticker_history_is_cut_at_the_datastream_delisting_date():
    sm = _derive(
        names=_names([(1.0, date(2007, 3, 1), OPEN, 7.0, "X", "DEAD CO", date(2018, 6, 16))]),
        tickers=_dated(
            [
                (1.0, date(2003, 1, 1), OPEN, "TWX"),
                (1.0, date(2022, 3, 1), date(2022, 3, 1), "JUNK"),
            ],
            "ticker",
        ),
    )
    t = _of(sm, "1", "ticker_datastream")
    assert t.rows() == [
        ("1", "ticker_datastream", "TWX", date(2003, 1, 1), date(2018, 6, 16), "datastream")
    ]


def test_permno_links_through_the_cusip_valid_on_both_sides_up_to_crsp_last():
    sm = _derive(
        cusips=_dated(
            [
                (1.0, date(1990, 1, 1), date(2013, 11, 19), "AAAA0001"),
                (1.0, date(2013, 11, 20), OPEN, "BBBB0002"),
            ],
            "cusip",
        ),
        crsp=_crsp(
            [
                (10, 100, date(2000, 1, 1), date(2013, 11, 30), "AAAA00019", "OLDT"),
                (11, 100, date(2013, 11, 20), CRSP_LAST, "BBBB00028", "NEWT"),
                (99, 900, date(2000, 1, 1), CRSP_LAST, "ZZZZ00009", "OTHR"),
            ]
        ),
    )
    p = _of(sm, "1", "permno")
    assert p.select("value", "valid_from", "valid_to").rows() == [
        ("10", date(2000, 1, 1), date(2013, 11, 19)),
        ("11", date(2013, 11, 20), CRSP_LAST),
    ]
    assert _of(sm, "1", "permco")["value"].to_list() == ["100"]  # merged: same value throughout


def test_merge_joins_adjacent_rows_with_the_same_value_only():
    rows = pl.DataFrame(
        [
            ("1", "cusip", "A", date(2000, 1, 1), date(2000, 12, 31), "x"),
            ("1", "cusip", "A", date(2001, 1, 1), date(2001, 6, 30), "x"),
            ("1", "cusip", "B", date(2001, 7, 1), None, "x"),
        ],
        schema=smod.SCHEMA,
        orient="row",
    )
    out = smod.merge(rows)
    assert out.select("value", "valid_from", "valid_to").rows() == [
        ("A", date(2000, 1, 1), date(2001, 6, 30)),
        ("B", date(2001, 7, 1), None),
    ]


def test_overrides_cut_derived_rows_and_an_empty_value_only_removes(tmp_path):
    sm = pl.DataFrame(
        [("1", "ticker_massive", "AUTO", date(2010, 1, 1), date(2010, 12, 31), "massive")],
        schema=smod.SCHEMA,
        orient="row",
    )
    path = tmp_path / "o.csv"
    path.write_text(
        "security_id,id_type,value,valid_from,valid_to,note\n"
        "1,ticker_massive,FIX,2010-03-01,2010-03-31,reviewed\n"
        "1,ticker_massive,,2010-06-01,2010-06-30,no symbol\n"
    )
    out = smod.apply_overrides(sm, smod.read_overrides(path))
    assert out.select("value", "valid_from", "valid_to", "source").rows() == [
        ("AUTO", date(2010, 1, 1), date(2010, 2, 28), "massive"),
        ("FIX", date(2010, 3, 1), date(2010, 3, 31), "override"),
        ("AUTO", date(2010, 4, 1), date(2010, 5, 31), "massive"),
        ("AUTO", date(2010, 7, 1), date(2010, 12, 31), "massive"),
    ]


def test_overrides_reject_unknown_id_types(tmp_path):
    path = tmp_path / "o.csv"
    path.write_text(
        "security_id,id_type,value,valid_from,valid_to,note\n1,nickname,X,2010-01-01,,x\n"
    )
    with pytest.raises(ValueError, match="unknown id_type"):
        smod.read_overrides(path)


def test_violations_name_securities_and_dates_never_values():
    sm = pl.DataFrame(
        [
            ("1", "permno", "123", date(2010, 1, 1), date(2010, 12, 31), "crsp"),
            ("1", "permno", "456", date(2010, 6, 1), None, "crsp"),  # two values on a date
            ("2", "ticker_massive", "FOXA", date(2019, 1, 1), date(2019, 3, 20), "massive"),
            ("3", "ticker_massive", "FOXA", date(2019, 3, 19), None, "massive"),  # shared
            ("4", "company", "9", date(2010, 1, 1), None, "datastream"),
            ("5", "company", "9", date(2010, 1, 1), None, "datastream"),  # may be shared
        ],
        schema=smod.SCHEMA,
        orient="row",
    )
    v = smod.violations(sm)
    assert v == [
        "security 1: two permno values on 2010-06-01",
        "securities 2 and 3 share a ticker_massive on 2019-03-19",
    ]
    assert not any(x in " ".join(v) for x in ("123", "456"))


def test_lookup_resolve_and_attach_use_the_row_valid_on_the_date():
    sm = pl.DataFrame(
        [
            ("1", "ticker_massive", "FB", date(2012, 5, 18), date(2022, 6, 8), "massive"),
            ("1", "ticker_massive", "META", date(2022, 6, 9), None, "massive"),
            ("2", "ticker_massive", "META", date(2021, 1, 1), date(2022, 6, 8), "massive"),
        ],
        schema=smod.SCHEMA,
        orient="row",
    )
    assert smod.lookup(sm, ["1", "2", "3"], "ticker_massive", date(2022, 6, 8)).to_list() == [
        "FB",
        "META",
        None,
    ]
    assert smod.resolve(sm, "ticker_massive", ["META", "FB"], date(2022, 6, 9)).to_list() == [
        "1",
        None,
    ]
    df = pl.DataFrame(
        {"ticker": ["1", "1", "2"], "day": [date(2020, 1, 2), date(2023, 1, 3), date(2021, 5, 3)]}
    )
    out = smod.attach(df, sm, "ticker_massive", key="ticker", date="day")
    assert out["ticker_massive"].to_list() == ["FB", "META", "META"]
    assert out.columns == ["ticker", "day", "ticker_massive"]
