"""Adjust a membership universe's raw bars with CRSP's distribution factors.

Contract: docs/CONTRACTS.md ("prices.parquet"). `gnns build-data` calls `build`, which
reads the raw bars and security master `gnns build-universe` wrote and the cached CRSP
extracts, and returns the prices and review reports unwritten. Nothing here touches the
network.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from gnns.data import security_master, store, universe, wrds
from gnns.data.security_master import FAR

log = logging.getLogger(__name__)
EXTRACTS = {
    "crsp_distributions": ["permno", "disexdt", "disdetailtype", "disfacshr"],
    "crsp_daily": ["permno", "dlycaldt", "dlyret", "dlyprc", "dlyprevprc"],
}
TERMINAL = {"SECMRG", "SECEX", "SECMWM"}  # distributions that end the security
STRAY = 1e-4  # a CRSP day factor this far from 1 without a distribution is reported
MATCH_TOLERANCES = (0.001, 0.005, 0.02)
REPORT_TOLERANCE = 0.02  # CRSP-match rows worse than this are written out
BAR_COLS = ["open", "high", "low", "close", "volume"]


@dataclass
class Built:
    prices: pl.DataFrame
    manifest: dict  # fields beyond built_at/git_sha
    reports: dict[str, pl.DataFrame] = field(default_factory=dict)
    summary: list[str] = field(default_factory=list)


def build(name: str, start: date, end: date, root: Path) -> Built:
    """A membership universe's prices; raises if `gnns build-universe` hasn't built its raw
    bars for this date range."""
    ex = {}
    for n, cols in EXTRACTS.items():
        df, _ = wrds.load(root, n)
        if missing := sorted(set(cols) - set(df.columns)):
            raise ValueError(f"WRDS extract {n} lacks {missing}; re-run gnns fetch-wrds")
        ex[n] = df
    dest = root / name / "raw_bars"
    if not (dest / "manifest.json").exists():
        raise FileNotFoundError(f"{dest} missing; run gnns build-universe first")
    lf, m = store.open(dest)
    if (m.get("start"), m.get("end")) != (str(start), str(end)):
        raise ValueError(
            f"{name}'s raw bars were built for {m.get('start')}..{m.get('end')}, not "
            f"{start}..{end}; run gnns build-universe"
        )
    raw = lf.collect().with_columns(date=pl.col("timestamp").dt.date())
    sm, _ = store.open(root / "security_master")
    table, _ = universe.load(root, name)
    return assemble(raw, sm.collect(), table, ex, m["tickers"], start, end)


def assemble(
    raw: pl.DataFrame,
    sm: pl.DataFrame,
    table: pl.DataFrame,
    ex: dict[str, pl.DataFrame],
    tickers: list[str],
    start: date,
    end: date,
) -> Built:
    crsp_last = ex["crsp_daily"]["dlycaldt"].max()
    assert isinstance(crsp_last, date)
    events, unpriced, stray = crsp_events(ex["crsp_distributions"], ex["crsp_daily"], sm)
    prices = adjust(raw, events)
    store.validate(prices, tickers, start, end)
    members = table.select("timestamp", ticker="security_id")
    shares, mismatches = crsp_match(prices, members, sm, ex["crsp_daily"], crsp_last)
    reports = {
        "distributions_unpriced": unpriced,
        "crsp_factor_without_distribution": stray,
        "crsp_return_mismatches": mismatches,
    }
    summary = [
        f"{len(tickers)} securities, {prices.height:,} bars, {events.height:,} CRSP "
        "distribution days applied",
        "CRSP match on member sessions: "
        + ", ".join(f"{k} {v:.4f}" for k, v in shares.items() if k != "compared")
        + f" of {int(shares['compared']):,}",
    ]
    summary += [f"report {k}: {v.height} rows" for k, v in reports.items() if v.height]
    manifest = {
        "source": "massive unadjusted bars x crsp distribution factors",
        "crsp_last": str(crsp_last),
        "tickers": tickers,
    }
    return Built(prices, manifest, reports, summary)


def crsp_events(
    dist: pl.DataFrame, daily: pl.DataFrame, sm: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """One row per (security_id, date) with a CRSP distribution ex-date of the PERMNO
    linked on that date: `factor`, CRSP's whole day `|price| / (|previous price| × (1 +
    total return))` (cash and stock dividends, splits, spinoffs and returns of capital
    alike, and several on one day together), and `split_factor`, the product of `1 / (1 +
    disfacshr)` over the day's distributions (the share part, for volume).

    Distributions that end the security (`SECMRG`, `SECEX`, `SECMWM`) are skipped: its
    delisting return is the final value. Also returns the distribution days CRSP has no
    usable day row for, and the linked days whose factor is off 1 by more than STRAY
    without a distribution (reported, never applied).
    """
    keep = ~pl.col("disdetailtype").is_in(list(TERMINAL)).fill_null(False)
    shr = pl.col("disfacshr").fill_null(0.0)
    per_day = (
        dist.filter(keep)
        .with_columns(permno=_permno("permno"), date=pl.col("disexdt"))
        .group_by("permno", "date")
        .agg(
            split_factor=pl.when(1 + shr > 0).then(1 / (1 + shr)).otherwise(1.0).product(),
        )
    )
    links = sm.filter(pl.col("id_type") == "permno").select(
        "security_id", permno="value", vf="valid_from", vt=pl.col("valid_to").fill_null(FAR)
    )
    day = daily.select(
        permno=_permno("permno"),
        date="dlycaldt",
        factor=pl.col("dlyprc").abs() / (pl.col("dlyprevprc").abs() * (1 + pl.col("dlyret"))),
    )
    ev = (
        per_day.join(links, on="permno")
        .filter(pl.col("date").is_between("vf", "vt"))
        .join(day, on=["permno", "date"], how="left")
    )
    ok = pl.col("factor").is_finite() & (pl.col("factor") > 0)
    unpriced = (
        ev.filter(~ok.fill_null(False)).select("security_id", "date").sort("security_id", "date")
    )
    events = (
        ev.filter(ok)
        .select("security_id", "date", "factor", "split_factor")
        .sort("security_id", "date")
    )
    stray = (
        day.join(links, on="permno")
        .filter(pl.col("date").is_between("vf", "vt"))
        .filter(((pl.col("factor") - 1).abs() > STRAY).fill_null(False))
        .join(per_day.select("permno", "date"), on=["permno", "date"], how="anti")
        .select("security_id", "date")
        .sort("security_id", "date")
    )
    return events, unpriced, stray


def _permno(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Int64).cast(pl.String)


def adjust(raw: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Unadjusted bars × the product of every later event's factor (OHLC); volume ÷ the
    product of later split factors. Returns the prices.parquet columns, keyed by
    security_id in `ticker`."""
    suffix = (
        events.sort("security_id", "date", descending=[False, True])
        .with_columns(
            f=pl.col("factor").cum_prod().over("security_id"),
            s=pl.col("split_factor").cum_prod().over("security_id"),
            key=pl.col("date") - timedelta(days=1),
        )
        .select("security_id", "key", "f", "s")
        .sort("key")
    )
    out = (
        raw.sort("date")
        .join_asof(
            suffix,
            left_on="date",
            right_on="key",
            by="security_id",
            strategy="forward",
            check_sortedness=False,
        )
        .with_columns(f=pl.col("f").fill_null(1.0), s=pl.col("s").fill_null(1.0))
        .with_columns(
            *(pl.col(c) * pl.col("f") for c in ("open", "high", "low", "close")),
            volume=pl.col("volume") / pl.col("s"),
        )
    )
    return out.select("timestamp", pl.col("security_id").alias("ticker"), *BAR_COLS).sort(
        "timestamp", "ticker"
    )


def crsp_match(
    prices: pl.DataFrame,
    members: pl.DataFrame,
    sm: pl.DataFrame,
    daily: pl.DataFrame,
    crsp_last: date,
) -> tuple[dict[str, float], pl.DataFrame]:
    """How often our adjusted close-to-close return matches CRSP's `dlyret` on member
    sessions CRSP covers (where the security also had a bar the session before).
    Returns aggregate shares and the rows off by more than REPORT_TOLERANCE."""
    axis = prices.select(pl.col("timestamp").unique().sort()).with_columns(
        prev_ts=pl.col("timestamp").shift(1)
    )
    p = prices.select("timestamp", "ticker", "close")
    r = (
        members.join(axis, on="timestamp")
        .join(p, on=["timestamp", "ticker"])
        .join(p.rename({"timestamp": "prev_ts", "close": "prev_close"}), on=["prev_ts", "ticker"])
        .with_columns(
            ret=pl.col("close") / pl.col("prev_close") - 1, day=pl.col("timestamp").dt.date()
        )
        .filter(pl.col("day") <= crsp_last)
    )
    r = (
        security_master.attach(r, sm, "permno", key="ticker", date="day")
        .join(
            daily.select(permno=_permno("permno"), day="dlycaldt", crsp_ret="dlyret"),
            on=["permno", "day"],
        )
        .drop_nulls("crsp_ret")
        .with_columns(diff=(pl.col("ret") - pl.col("crsp_ret")).abs())
    )
    shares = {
        f"within_{t}": r.select((pl.col("diff") <= t).mean()).item() for t in MATCH_TOLERANCES
    }
    shares["compared"] = float(r.height)
    bad = r.filter(pl.col("diff") > REPORT_TOLERANCE).select("ticker", "day", "ret", "crsp_ret")
    return shares, bad
