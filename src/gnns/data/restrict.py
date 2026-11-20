"""Restrictions: narrow a universe table (`gnns.data.universe`) to a subset, point in time.

Filters rank members only: `apply` first keeps the rows with `member` true, then runs each
filter, which takes the table and returns the rows it keeps. The columns it ranks on are
known at each row's `timestamp` (the session close), so a filter at `t` uses nothing after
`t`. `at` says when a filter decides:
- `"timestamp"`: every session, on that session's values;
- `"list_date"`: once per source list, on the values at the list date (a security's first
  row of the list's period), held until the next list.

A construction lists its filters in its graph file, applied in order:

    universe_filters:
      - one_class_per_company: {by: adv_63}
      - top_n: {by: mktcap, n: 100}
"""

import inspect
from collections.abc import Callable

import polars as pl

AT = ("timestamp", "list_date")


def _values(u: pl.DataFrame, by: str, at: str) -> pl.DataFrame:
    """(at, security_id, value): the `by` value each decision uses."""
    if by not in u.columns:
        raise ValueError(f"no column {by!r} to rank on; columns: {u.columns}")
    if at not in AT:
        raise ValueError(f"at must be one of {list(AT)}, got {at!r}")
    if at == "timestamp":
        return u.select(at, "security_id", value=by)
    return (
        u.sort("timestamp")
        .group_by(at, "security_id")
        .agg(value=pl.col(by).first(), company=pl.col("company").first())
    )


def _ranked(v: pl.DataFrame, group: list[str]) -> pl.DataFrame:
    """`v` plus `rank` (0 = best) within `group`: highest value first, nulls last, ties to
    the lower security_id."""
    return v.sort(
        [*group, "value", "security_id"],
        descending=[False] * len(group) + [True, False],
        nulls_last=True,
    ).with_columns(rank=pl.int_range(pl.len()).over(group))


def one_class_per_company(
    u: pl.DataFrame, by: str = "adv_63", at: str = "list_date"
) -> pl.DataFrame:
    """Keep one security per company (`company`; a security without one is its own): the
    one with the highest `by`."""
    v = _values(u, by, at)
    if "company" not in v.columns:
        v = v.join(
            u.select(at, "security_id", "company").unique([at, "security_id"]),
            on=[at, "security_id"],
        )
    v = v.with_columns(company=pl.coalesce("company", "security_id"))
    keep = _ranked(v, [at, "company"]).filter(pl.col("rank") == 0)
    return u.join(keep.select(at, "security_id"), on=[at, "security_id"], how="semi")


def top_n(u: pl.DataFrame, by: str, n: int, at: str = "timestamp") -> pl.DataFrame:
    """Keep the `n` securities with the highest `by` (a null value is never kept)."""
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"top_n: n must be a positive integer, got {n!r}")
    v = _values(u, by, at).drop_nulls("value")
    keep = _ranked(v, [at]).filter(pl.col("rank") < n)
    return u.join(keep.select(at, "security_id"), on=[at, "security_id"], how="semi")


FILTERS: dict[str, Callable[..., pl.DataFrame]] = {
    "one_class_per_company": one_class_per_company,
    "top_n": top_n,
}


def _parse(filters: list[dict]) -> list[tuple[str, dict]]:
    out = []
    for f in filters:
        if not isinstance(f, dict) or len(f) != 1:
            raise ValueError(f"a universe filter is one {{name: {{arguments}}}} entry, got {f!r}")
        ((name, kwargs),) = f.items()
        if name not in FILTERS:
            raise ValueError(f"unknown universe filter {name!r}; valid: {sorted(FILTERS)}")
        kwargs = kwargs or {}
        params = inspect.signature(FILTERS[name]).parameters
        if unknown := sorted(set(kwargs) - set(params) - {"u"}):
            raise ValueError(f"universe filter {name}: unknown arguments {unknown}")
        out.append((name, kwargs))
    return out


def check(filters: list[dict]) -> None:
    """Raise if `filters` (a graph file's `universe_filters`) name an unknown filter or
    argument; reads no data."""
    _parse(filters)


def apply(u: pl.DataFrame, filters: list[dict]) -> pl.DataFrame:
    """The member rows of `u` (all rows if it has no `member` column) after each filter in
    order."""
    if "member" in u.columns:
        u = u.filter("member")
    for name, kwargs in _parse(filters):
        u = FILTERS[name](u, **kwargs)
    return u
