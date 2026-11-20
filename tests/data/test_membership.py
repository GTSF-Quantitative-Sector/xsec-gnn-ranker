"""Membership universes: month-end lists -> intervals -> member sessions."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import polars as pl

from gnns.data import membership

NY = ZoneInfo("America/New_York")
J, F, M, A = date(2020, 1, 31), date(2020, 2, 28), date(2020, 3, 31), date(2020, 4, 30)


def _close(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 16, tzinfo=NY)


def _spells(rows):
    return pl.DataFrame(
        [(4408.0, float(i), s, e) for i, s, e in rows],
        schema={
            "indexlistintcode": pl.Float64,
            "infocode": pl.Float64,
            "startdate": pl.Date,
            "enddate": pl.Date,
        },
        orient="row",
    )


SPELLS = _spells([(1, J, A), (2, J, F), (3, M, A)])  # 2 leaves after Feb, 3 joins in Mar


def test_list_members_start_at_the_last_list_on_or_before_start():
    m = membership.list_members(SPELLS, 4408, date(2020, 2, 15), date(2020, 3, 31))
    assert m.rows() == [(J, "1"), (J, "2"), (F, "1"), (F, "2"), (M, "1"), (M, "3")]


def test_intervals_run_from_first_list_to_the_next_list_not_on_and_latest_is_open():
    m = membership.list_members(SPELLS, 4408, J, A)
    iv = membership.intervals(m, m["list_date"])
    assert iv.rows() == [
        ("1", _close(J), None),
        ("2", _close(J), _close(M)),
        ("3", _close(M), None),
    ]
    assert membership.violations(iv) == []


def test_a_gap_between_lists_gives_two_intervals():
    m = membership.list_members(_spells([(1, J, J), (1, M, A), (2, J, F), (3, M, A)]), 4408, J, A)
    iv = membership.intervals(m, m["list_date"]).filter(pl.col("security_id") == "1")
    assert iv.rows() == [("1", _close(J), _close(F)), ("1", _close(M), None)]


def test_member_rows_include_start_and_exclude_end():
    iv = pl.DataFrame(
        {"security_id": ["2"], "start": [_close(J)], "end": [_close(M)]},
        schema=membership.SCHEMA,
    )
    ts = pl.Series([_close(J), _close(F), _close(M)])
    assert membership.member_rows(iv, ts)["timestamp"].to_list() == [_close(J), _close(F)]


def test_membership_at_t_uses_only_lists_known_by_t():
    """Look-ahead: cutting the source after list date d changes no membership at t <= d."""
    full = _spells([(1, J, A), (2, J, F), (3, M, A), (4, F, M)])
    known_by_f = full.filter(pl.col("startdate") <= F).with_columns(
        enddate=pl.min_horizontal("enddate", pl.lit(F))
    )
    ts = pl.Series([_close(J), _close(date(2020, 2, 14)), _close(F)])

    def mask(spells):
        m = membership.list_members(spells, 4408, J, A)
        return membership.member_rows(membership.intervals(m, m["list_date"]), ts)

    assert mask(full).equals(mask(known_by_f))
