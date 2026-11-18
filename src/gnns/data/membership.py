"""Membership universes: who is a member when. Contract: docs/CONTRACTS.md,
"Membership universes".

A source list dated `d` (a month-end trading day) is known at `d` 16:00 ET. A security is
a member at `t` when `start <= t < end`: `start` is 16:00 on the first list date of a run
of consecutive lists it is on, `end` 16:00 on the next list date it is not on (null if it
is on the latest list).
"""

from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from gnns.data import store
from gnns.data.security_master import security_ids

TZ = "America/New_York"
SCHEMA = {
    "security_id": pl.String,
    "start": pl.Datetime("us", TZ),
    "end": pl.Datetime("us", TZ),
}


def at_close(d: date) -> datetime:
    return datetime.combine(d, time(16), tzinfo=ZoneInfo(TZ))


def list_members(
    constituents: pl.DataFrame, list_code: int, start: date, end: date
) -> pl.DataFrame:
    """(list_date, security_id) for every list in range: the last list on or before
    `start` through the last list on or before `end`.

    `constituents` is Datastream's spell table (`infocode, startdate, enddate`, plus
    `indexlistintcode`); its spells run from the first to the last list a security is on.
    List dates are the spells' start and end dates: a month-end on which nothing changed
    is not one, and membership simply carries over it (a removal always lands on a list
    date, because its replacement's spell starts there).
    """
    spells = constituents.filter(
        (pl.col("indexlistintcode") == list_code) & pl.col("infocode").is_not_null()
    ).select(security_id=security_ids(pl.col("infocode")), s="startdate", e="enddate")
    dates = pl.concat([spells["s"], spells["e"]]).unique().sort()
    first = dates.filter(dates <= start).max()
    in_range = dates.filter((dates >= (first or dates.min())) & (dates <= end))
    return (
        pl.DataFrame({"list_date": in_range})
        .join(spells, how="cross")
        .filter(pl.col("list_date").is_between("s", "e"))
        .select("list_date", "security_id")
        .unique()
        .sort("list_date", "security_id")
    )


def keep_one_class(
    members: pl.DataFrame, company: pl.DataFrame, liquidity: pl.DataFrame, window: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Keep one security per company on each list: the highest median unadjusted
    close × volume over the `window` bars ending at the list date.

    `company`: security-master rows of id_type `company`. `liquidity`:
    `(date, security_id, dollar_volume)` from unadjusted bars. Returns the kept members
    and the dropped `(list_date, security_id, kept)` rows.
    """
    med = (
        liquidity.sort("security_id", "date")
        .with_columns(
            med=pl.col("dollar_volume").rolling_median(window, min_samples=1).over("security_id")
        )
        .select("security_id", "date", "med")
    )
    comp = company.select("security_id", co="value", f="valid_from", t="valid_to")
    m = (
        members.join(comp, on="security_id", how="left")
        .filter(
            pl.col("co").is_null()
            | (
                (pl.col("f") <= pl.col("list_date"))
                & (pl.col("t").is_null() | (pl.col("t") >= pl.col("list_date")))
            )
        )
        .unique(["list_date", "security_id"], keep="first")
        .with_columns(co=pl.col("co").fill_null(pl.col("security_id")))
        .sort("list_date")
        .join_asof(
            med.sort("date"),
            left_on="list_date",
            right_on="date",
            by="security_id",
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns(
            rank=pl.struct(pl.col("med").fill_null(-1.0), "security_id")
            .rank("ordinal", descending=True)
            .over("list_date", "co")
        )
    )
    kept = m.filter(pl.col("rank") == 1).select("list_date", "security_id")
    winner = kept.join(m.select("list_date", "security_id", "co"), on=["list_date", "security_id"])
    dropped = (
        m.filter(pl.col("rank") > 1)
        .select("list_date", "security_id", "co")
        .join(winner.rename({"security_id": "kept"}), on=["list_date", "co"])
        .select("list_date", "security_id", "kept")
    )
    return kept.sort("list_date", "security_id"), dropped.sort("list_date", "security_id")


def company_duplicates(members: pl.DataFrame, company: pl.DataFrame) -> pl.DataFrame:
    """(list_date, co) for every list on which two members share a company code."""
    comp = company.select("security_id", co="value", f="valid_from", t="valid_to")
    return (
        members.join(comp, on="security_id")
        .filter(
            (pl.col("f") <= pl.col("list_date"))
            & (pl.col("t").is_null() | (pl.col("t") >= pl.col("list_date")))
        )
        .group_by("list_date", "co")
        .agg(n=pl.col("security_id").n_unique())
        .filter(pl.col("n") > 1)
        .select("list_date", "co")
        .sort("list_date", "co")
    )


def intervals(members: pl.DataFrame, list_dates: pl.Series) -> pl.DataFrame:
    """Membership intervals from per-list members; `list_dates` are every list date in
    the source (so a run's end is the next list, even past the members' range)."""
    dates = list_dates.unique().sort()
    pos = pl.DataFrame({"list_date": dates}).with_columns(
        i=pl.int_range(pl.len()),
        ts=closes(pl.col("list_date")),
    )
    runs = (
        members.join(pos.select("list_date", "i"), on="list_date")
        .sort("security_id", "i")
        .with_columns(run=(pl.col("i").diff().fill_null(1) != 1).cum_sum().over("security_id"))
        .group_by("security_id", "run")
        .agg(first=pl.col("i").min(), next=pl.col("i").max() + 1)
    )
    return (
        runs.join(pos.select(first="i", start="ts"), on="first")
        .join(pos.select(next="i", end="ts"), on="next", how="left")
        .select(pl.col(c).cast(t) for c, t in SCHEMA.items())
        .sort("security_id", "start")
    )


def closes(day: pl.Expr) -> pl.Expr:
    """Date -> 16:00 America/New_York on that date."""
    return day.dt.combine(time(16)).dt.replace_time_zone(TZ)


def violations(iv: pl.DataFrame) -> list[str]:
    """Broken invariants of an intervals table: start < end, no overlaps per security."""
    out = [
        f"security {r['security_id']}: interval starting {r['start']} does not end after it"
        for r in iv.filter(
            pl.col("end").is_not_null() & (pl.col("end") <= pl.col("start"))
        ).iter_rows(named=True)
    ]
    over = (
        iv.sort("security_id", "start")
        .with_columns(prev=pl.col("end").shift(1).over("security_id"))
        .filter(pl.col("prev").is_not_null() & (pl.col("start") < pl.col("prev")))
    )
    out += [
        f"security {r['security_id']}: overlapping intervals at {r['start']}"
        for r in over.iter_rows(named=True)
    ]
    return out


def member_rows(iv: pl.DataFrame, timestamps: pl.Series) -> pl.DataFrame:
    """(timestamp, ticker) for every timestamp in `timestamps` at which each security
    is a member."""
    ts = pl.DataFrame({"timestamp": timestamps.unique().sort()})
    return (
        ts.join(iv, how="cross")
        .filter(
            (pl.col("start") <= pl.col("timestamp"))
            & (pl.col("end").is_null() | (pl.col("timestamp") < pl.col("end")))
        )
        .select("timestamp", ticker="security_id")
        .unique()
        .sort("timestamp", "ticker")
    )


def mask(root: Path, universe: str, axis: pl.Series) -> pl.DataFrame:
    """(timestamp, ticker) for every bar-axis timestamp at which each security is a
    member; empty for list universes, which have no membership dataset."""
    dest = root / universe / "membership"
    if not (dest / "manifest.json").exists():
        return pl.DataFrame(schema={"timestamp": axis.dtype, "ticker": pl.String})
    lf, _ = store.open(dest)
    return member_rows(lf.collect(), axis)


def exists(root: Path, universe: str) -> bool:
    return (root / universe / "membership" / "manifest.json").exists()
