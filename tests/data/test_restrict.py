"""Restrictions on a universe table: one class per company, top n by any column."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from gnns.data import restrict

NY = ZoneInfo("America/New_York")
J, F = date(2020, 1, 31), date(2020, 2, 28)


def _ts(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 16, tzinfo=NY)


def _table(rows) -> pl.DataFrame:
    """(day, security_id, list_date, company, adv_63, mktcap)."""
    return pl.DataFrame(
        [(_ts(d), s, ld, co, adv, cap) for d, s, ld, co, adv, cap in rows],
        schema={
            "timestamp": pl.Datetime("us", "America/New_York"),
            "security_id": pl.String,
            "list_date": pl.Date,
            "company": pl.String,
            "adv_63": pl.Float64,
            "mktcap": pl.Float64,
        },
        orient="row",
    )


D1, D2 = date(2020, 2, 3), date(2020, 2, 4)
# GOOG-like pair (A, B: company 9) whose liquidity crosses inside January's list period.
U = _table(
    [
        (J, "A", J, "9", 5.0, 100.0),
        (J, "B", J, "9", 4.0, 100.0),
        (J, "C", J, None, 1.0, 50.0),
        (D1, "A", J, "9", 3.0, 100.0),
        (D1, "B", J, "9", 6.0, 100.0),
        (D1, "C", J, None, 1.0, 50.0),
        (F, "A", F, "9", 3.0, 100.0),
        (F, "B", F, "9", 6.0, 100.0),
        (F, "C", F, None, None, 50.0),
    ]
)


def _on(u: pl.DataFrame) -> dict[date, list[str]]:
    return {
        d: sorted(g["security_id"].to_list())
        for (d,), g in u.group_by(pl.col("timestamp").dt.date(), maintain_order=True)
    }


def test_one_class_per_company_decides_at_each_list_and_holds():
    out = restrict.one_class_per_company(U)
    # January's list keeps A (its adv on the list date) for the whole period, even after
    # B overtakes it; February's list switches to B. C has no company: it is its own.
    assert _on(out) == {J: ["A", "C"], D1: ["A", "C"], F: ["B", "C"]}


def test_one_class_per_company_can_decide_every_session():
    out = restrict.one_class_per_company(U, at="timestamp")
    assert _on(out)[D1] == ["B", "C"]


def test_top_n_ranks_on_any_column_and_never_keeps_a_null():
    assert _on(restrict.top_n(U, by="adv_63", n=1)) == {J: ["A"], D1: ["B"], F: ["B"]}
    assert _on(restrict.top_n(U, by="adv_63", n=3))[F] == ["A", "B"]  # C's adv is null
    # mktcap ties between A and B go to the lower security id.
    assert _on(restrict.top_n(U, by="mktcap", n=1))[J] == ["A"]


def test_filters_apply_in_order():
    out = restrict.apply(
        U, [{"one_class_per_company": {"by": "adv_63"}}, {"top_n": {"by": "mktcap", "n": 2}}]
    )
    assert _on(out) == {J: ["A", "C"], D1: ["A", "C"], F: ["B", "C"]}


def test_a_filter_at_t_uses_nothing_after_t():
    """Look-ahead: dropping every row after D1 changes no kept row at t <= D1."""
    filters = [{"one_class_per_company": {}}, {"top_n": {"by": "adv_63", "n": 1}}]
    upto = U.filter(pl.col("timestamp").dt.date() <= D1)
    full = restrict.apply(U, filters).filter(pl.col("timestamp").dt.date() <= D1)
    assert (
        restrict.apply(upto, filters)
        .sort("timestamp", "security_id")
        .equals(full.sort("timestamp", "security_id"))
    )


@pytest.mark.parametrize(
    "filters, match",
    [
        ([{"bottom_n": {}}], "unknown universe filter 'bottom_n'"),
        ([{"top_n": {"by": "adv_63", "n": 5, "k": 1}}], r"unknown arguments \['k'\]"),
        ([{"top_n": {}, "one_class_per_company": {}}], "one {name: {arguments}} entry"),
    ],
)
def test_bad_filters_raise_before_any_data(filters, match):
    with pytest.raises(ValueError, match=match):
        restrict.check(filters)


def test_ranking_on_a_missing_column_names_the_columns():
    with pytest.raises(ValueError, match="no column 'adv_20'"):
        restrict.top_n(U, by="adv_20", n=1)
