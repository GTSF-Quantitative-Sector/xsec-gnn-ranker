"""Build a membership universe (the central `sp500`): security master, membership,
adjusted prices and delistings, from the cached WRDS extracts and Massive's bars.

Contract: docs/CONTRACTS.md ("Membership universes", "Security master", "prices.parquet").
`gnns build-data` calls `build`, which fetches from Massive, validates, and returns the
datasets and review reports unwritten. The steps after the Massive fetch are pure
functions of their inputs (tested on fixtures).
"""

import logging
import math
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from gnns.data import massive, membership, security_master, store, wrds
from gnns.data.security_master import FAR

log = logging.getLogger(__name__)
TZ = "America/New_York"
EXTRACTS = (
    "ds_constituents",
    "ds_tickers",
    "ds_cusips",
    "ds_names",
    "crsp_security",
    "crsp_distributions",
    "crsp_delistings",
    "crsp_daily",
)
FETCH_PAD_DAYS = 10  # extra days of bars around each symbol's window
BOUNDARY_DAYS = 7  # around a ticker change, the neighbouring symbol may carry the bars
MAX_SEAM = 0.2  # a borrowed symbol's close must be within this |log move| of the last close
MAX_SYMBOL_GAP = 5  # member sessions in a row without a symbol before the build fails
WORKERS = 8
JUMP = 0.30  # an unexplained |close-to-close| move above this goes to the review report
MATCH_TOLERANCES = (0.001, 0.005, 0.02)
REPORT_TOLERANCE = 0.02  # CRSP-match rows worse than this are written out
TERMINAL = {"SECMRG", "SECEX", "SECMWM"}  # CRSP "SP" distributions that end the security
CLASS_MARK = re.compile(r"'[A-Z]'")  # Datastream marks share classes like ALPHABET 'A'
DELIST_WINDOW = (timedelta(days=-10), timedelta(days=60))  # CRSP delisting vs last bar
BAR_COLS = ["open", "high", "low", "close", "volume"]


@dataclass
class Built:
    """Everything a membership universe's build produces, unwritten."""

    prices: pl.DataFrame
    membership: pl.DataFrame
    security_master: pl.DataFrame
    delistings: pl.DataFrame
    manifests: dict[str, dict]  # dataset -> manifest fields beyond built_at/git_sha
    reports: dict[str, pl.DataFrame] = field(default_factory=dict)
    summary: list[str] = field(default_factory=list)


# --- Massive -------------------------------------------------------------------------------


@dataclass
class MassiveData:
    bars: pl.DataFrame  # symbol, date, timestamp, open, high, low, close, volume (unadjusted)
    splits: pl.DataFrame  # symbol, date, split_from, split_to
    dividends: pl.DataFrame  # symbol, date, cash_amount, currency


def _parallel(fn: Callable, items: list) -> list:
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(fn, items))


def fetch_bars(windows: pl.DataFrame) -> pl.DataFrame:
    """Unadjusted daily bars for each `(symbol, start, end)` window."""

    def one(row: tuple) -> pl.DataFrame:
        sym, lo, hi = row
        return massive.daily_bars(sym, lo, hi, adjusted=False)

    rows = list(windows.select("symbol", "start", "end").iter_rows())
    frames = [f for f in _parallel(one, rows) if f.height]
    if not frames:
        return _empty_bars()
    return (
        pl.concat(frames)
        .rename({"ticker": "symbol"})
        .with_columns(date=pl.col("timestamp").dt.date())
        .unique(["symbol", "date"], keep="first")
    )


def fetch_events(symbols: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Every split and cash-dividend record of each symbol."""
    splits = [f for f in _parallel(massive.splits, symbols) if f.height]
    divs = [f for f in _parallel(massive.cash_dividends, symbols) if f.height]
    sp = pl.concat(splits).rename({"ticker": "symbol"}) if splits else _empty_splits()
    dv = pl.concat(divs).rename({"ticker": "symbol"}) if divs else _empty_dividends()
    return sp, dv


def _empty_bars() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("us", TZ),
            "symbol": pl.String,
            **dict.fromkeys(BAR_COLS, pl.Float64),
            "date": pl.Date,
        }
    )


def _empty_splits() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "symbol": pl.String,
            "date": pl.Date,
            "split_from": pl.Float64,
            "split_to": pl.Float64,
        }
    )


def _empty_dividends() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "symbol": pl.String,
            "date": pl.Date,
            "cash_amount": pl.Float64,
            "currency": pl.String,
            "distribution_type": pl.String,
        }
    )


# --- Symbols: which Massive ticker carries each security's bars on each session ------------


def ticker_windows(
    sm: pl.DataFrame, ds_names: pl.DataFrame, ids: list[str], start: date, end: date
) -> pl.DataFrame:
    """Each security's Datastream ticker intervals in [start, end], numbered `k` in date
    order, with the Massive spellings to try (`candidates`)."""
    classes = (
        ds_names.select(
            security_id=security_master.security_ids(pl.col("infocode")),
            cls=pl.col("dsqtname").str.contains(CLASS_MARK.pattern),
        )
        .group_by("security_id")
        .agg(pl.col("cls").any())
    )
    t = (
        sm.filter((pl.col("id_type") == "ticker_datastream") & pl.col("security_id").is_in(ids))
        .select(
            "security_id", ticker="value", vf="valid_from", vt=pl.col("valid_to").fill_null(FAR)
        )
        .filter((pl.col("vt") >= start) & (pl.col("vf") <= end))
        .with_columns(
            vf=pl.max_horizontal("vf", pl.lit(start)), vt=pl.min_horizontal("vt", pl.lit(end))
        )
        .join(classes, on="security_id", how="left")
        .sort("security_id", "vf")
        .with_columns(k=pl.int_range(pl.len()).over("security_id"))
    )
    return t.with_columns(
        candidates=pl.struct("ticker", "cls").map_elements(
            lambda r: spellings(r["ticker"], bool(r["cls"])), return_dtype=pl.List(pl.String)
        )
    )


def spellings(ticker: str, has_class: bool) -> list[str]:
    """Massive spellings to try for a Datastream ticker: as written, plus `BRK.B` for a
    class share written `BRKB`, or `BRKB` for one written `BRK.B`."""
    out = [ticker]
    if "." in ticker:
        out.append(ticker.replace(".", ""))
    elif has_class and len(ticker) >= 2:
        out.append(f"{ticker[:-1]}.{ticker[-1]}")
    return out


def fetch_windows(tw: pl.DataFrame) -> pl.DataFrame:
    """(symbol, start, end): the date range to fetch for each candidate spelling."""
    pad = timedelta(days=FETCH_PAD_DAYS)
    return (
        tw.explode("candidates")
        .group_by(symbol="candidates")
        .agg(start=pl.col("vf").min() - pad, end=pl.col("vt").max() + pad)
        .sort("symbol")
    )


def choose_symbols(tw: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    """Each ticker interval's Massive symbol: the candidate with the most bars inside the
    interval (null if none has any)."""
    counts = (
        tw.select("security_id", "k", "vf", "vt", symbol="candidates")
        .explode("symbol")
        .with_columns(order=pl.int_range(pl.len()).over("security_id", "k"))
        .join(bars.select("symbol", "date"), on="symbol", how="left")
        .group_by("security_id", "k", "symbol", "order")
        .agg(n=(pl.col("date").is_between(pl.col("vf"), pl.col("vt"))).sum())
        .filter(pl.col("n") > 0)
        .sort("security_id", "k", pl.col("n"), "order", descending=[False, False, True, False])
        .unique(["security_id", "k"], keep="first")
        .select("security_id", "k", "symbol")
    )
    return tw.select("security_id", "k", "ticker", "vf", "vt").join(
        counts, on=["security_id", "k"], how="left"
    )


def assign_symbols(
    intervals: pl.DataFrame, bars: pl.DataFrame, sessions: pl.Series
) -> pl.DataFrame:
    """(security_id, date, symbol) for every session inside a ticker interval.

    Away from a ticker change the interval's own symbol is used. Within BOUNDARY_DAYS of a
    change, the neighbouring interval's symbol is a candidate too, because Datastream
    sometimes switches a day early or late (`_resolve_boundaries` decides those days).
    """
    iv = intervals.sort("security_id", "k").with_columns(
        prev_sym=pl.col("symbol").shift(1).over("security_id"),
        next_sym=pl.col("symbol").shift(-1).over("security_id"),
        next_vf=pl.col("vf").shift(-1).over("security_id"),
    )
    days = pl.DataFrame({"date": sessions.unique().sort()})
    a = (
        days.join(iv, how="cross")
        .filter(pl.col("date").is_between("vf", "vt"))
        .with_columns(
            near_prev=pl.col("prev_sym").is_not_null()
            & ((pl.col("date") - pl.col("vf")).dt.total_days() < BOUNDARY_DAYS),
            near_next=pl.col("next_sym").is_not_null()
            & ((pl.col("next_vf") - pl.col("date")).dt.total_days() <= BOUNDARY_DAYS),
        )
    )
    close = bars.select("symbol", "date", "close")

    def with_close(col: str, name: str) -> pl.DataFrame:
        return close.rename({"symbol": col, "close": name})

    a = (
        a.join(with_close("symbol", "c_cur"), on=["symbol", "date"], how="left")
        .join(with_close("prev_sym", "c_prev"), on=["prev_sym", "date"], how="left")
        .join(with_close("next_sym", "c_next"), on=["next_sym", "date"], how="left")
        .sort("security_id", "date")
    )
    hard = (pl.col("near_prev") & pl.col("c_prev").is_not_null()) | (
        pl.col("near_next") & pl.col("c_next").is_not_null()
    )
    out = a.filter(~hard).select("security_id", "date", "symbol")
    if a.filter(hard).height:
        out = pl.concat([out, _resolve_boundaries(a.with_columns(hard=hard))])
    return out.sort("security_id", "date")


def _resolve_boundaries(a: pl.DataFrame) -> pl.DataFrame:
    """Boundary sessions where a neighbouring symbol has a bar, decided in date order.

    A neighbouring symbol is only borrowed if its close continues the security's previous
    chosen close (|log move| <= MAX_SEAM): at a handover the old or new symbol may belong
    to another security that day. Among the remaining candidates with a bar, the nearest
    close wins and the interval's own symbol wins ties (within 0.02); with none, the own
    symbol stands (no bar that session).
    """
    last_close: dict[str, float] = {}
    chosen = []
    for r in a.iter_rows(named=True):
        sid, d, own = r["security_id"], r["date"], r["symbol"]
        if not r["hard"]:
            if r["c_cur"] is not None:
                last_close[sid] = r["c_cur"]
            continue
        options: list[tuple[str, float]] = []
        if own is not None and r["c_cur"] is not None:
            options.append((own, r["c_cur"]))
        if r["near_prev"] and r["c_prev"] is not None:
            options.append((r["prev_sym"], r["c_prev"]))
        if r["near_next"] and r["c_next"] is not None:
            options.append((r["next_sym"], r["c_next"]))
        prev = last_close.get(sid)
        if prev is not None:
            options = [
                (s, c)
                for s, c in options
                if s == own or (c > 0 and abs(math.log(c / prev)) <= MAX_SEAM)
            ]
        if not options:
            pick = own
        elif prev is None:
            pick = own if own in dict(options) else options[0][0]
        else:
            dist = {s: abs(math.log(c / prev)) for s, c in options if c > 0}
            best = min(dist, key=lambda s: dist[s])
            pick = own if own in dist and dist[own] <= dist[best] + 0.02 else best
        chosen.append((sid, d, pick))
        c = dict(options).get(pick) if pick is not None else None
        if c is not None:
            last_close[sid] = c
    return pl.DataFrame(
        chosen,
        schema={"security_id": pl.String, "date": pl.Date, "symbol": pl.String},
        orient="row",
    )


def runs(assigned: pl.DataFrame) -> pl.DataFrame:
    """Session-by-session symbols -> security-master rows of id_type `ticker_massive`."""
    a = assigned.drop_nulls("symbol").sort("security_id", "date")
    return (
        a.with_columns(
            run=(pl.col("symbol") != pl.col("symbol").shift(1))
            .fill_null(True)
            .cum_sum()
            .over("security_id")
        )
        .group_by("security_id", "run", "symbol")
        .agg(valid_from=pl.col("date").min(), valid_to=pl.col("date").max())
        .select(
            "security_id",
            id_type=pl.lit("ticker_massive"),
            value="symbol",
            valid_from="valid_from",
            valid_to="valid_to",
            source=pl.lit("massive"),
        )
        .sort("security_id", "valid_from")
    )


def sessions_for(sm: pl.DataFrame, sessions: pl.Series) -> pl.DataFrame:
    """(security_id, date, symbol) from the final `ticker_massive` rows."""
    tm = sm.filter(pl.col("id_type") == "ticker_massive").select(
        "security_id", symbol="value", vf="valid_from", vt=pl.col("valid_to").fill_null(FAR)
    )
    return (
        pl.DataFrame({"date": sessions.unique().sort()})
        .join(tm, how="cross")
        .filter(pl.col("date").is_between("vf", "vt"))
        .select("security_id", "date", "symbol")
    )


# --- Adjustment ---------------------------------------------------------------------------


def split_events(splits: pl.DataFrame, used: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Massive split records of the symbol a security used around each date ->
    `(security_id, date, factor, kind)`; plus the records whose ratio is not round."""
    ev = _own_events(splits, used).select(
        "security_id",
        "date",
        factor=pl.col("split_from") / pl.col("split_to"),
        kind=pl.lit("split"),
        split_from="split_from",
        split_to="split_to",
        symbol="symbol",
    )
    ratio = pl.max_horizontal(
        pl.col("split_to") / pl.col("split_from"), pl.col("split_from") / pl.col("split_to")
    )
    odd = ev.filter(((ratio * 2) - (ratio * 2).round()).abs() > 1e-6)
    return ev.select("security_id", "date", "factor", "kind"), odd


def dividend_events(
    dividends: pl.DataFrame, used: pl.DataFrame, raw: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Massive cash dividends -> `(security_id, date, factor, kind)` with factor
    `1 - cash / unadjusted close of the bar before the ex-date`; plus the records skipped
    (non-USD, no earlier bar, or cash not below that close)."""
    own = _own_events(
        dividends.filter(pl.col("currency").is_null() | (pl.col("currency") == "USD")), used
    ).filter(pl.col("cash_amount") > 0)
    prev = raw.select("security_id", "date", prev_close="close").sort("date")
    ev = (
        own.sort("date")
        .join_asof(
            prev.with_columns(pl.col("date").alias("bar_date")),
            on="date",
            by="security_id",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .with_columns(factor=1 - pl.col("cash_amount") / pl.col("prev_close"))
    )
    ok = pl.col("factor").is_not_null() & (pl.col("factor") > 0) & (pl.col("factor") < 1)
    skipped = pl.concat(
        [
            ev.filter(~ok.fill_null(False)).select(
                "security_id", "date", "symbol", "cash_amount", "prev_close"
            ),
            _own_events(
                dividends.filter(pl.col("currency").is_not_null() & (pl.col("currency") != "USD")),
                used,
            )
            .select("security_id", "date", "symbol", "cash_amount")
            .with_columns(prev_close=pl.lit(None, pl.Float64)),
        ]
    )
    return ev.filter(ok).select("security_id", "date", "factor", kind=pl.lit("dividend")), skipped


def _own_events(events: pl.DataFrame, used: pl.DataFrame) -> pl.DataFrame:
    """Keep each symbol's records dated within a range of dates a security used it."""
    spans = used.group_by("security_id", "symbol").agg(
        lo=pl.col("date").min(), hi=pl.col("date").max()
    )
    return events.join(spans, on="symbol").filter(pl.col("date").is_between("lo", "hi"))


def spinoff_events(
    dist: pl.DataFrame, daily: pl.DataFrame, sm: pl.DataFrame, crsp_last: date
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """CRSP spinoffs (distribution type SP, not a terminal merger or exchange) up to
    `crsp_last` -> `(security_id, date, factor, kind)`; the factor is CRSP's for the whole
    ex-date, `|price| / (|previous price| × (1 + total return))`. Also returns the spinoffs
    that could not be priced (no linked security or no CRSP day row)."""
    sp = (
        dist.filter(
            (pl.col("distype") == "SP")
            & ~pl.col("disdetailtype").is_in(list(TERMINAL))
            & (pl.col("disexdt") <= crsp_last)
        )
        .select("permno", date="disexdt")
        .unique()
        .with_columns(permno=pl.col("permno").cast(pl.String))
    )
    links = sm.filter(pl.col("id_type") == "permno").select(
        "security_id", permno="value", vf="valid_from", vt=pl.col("valid_to").fill_null(FAR)
    )
    day = daily.select(
        permno=pl.col("permno").cast(pl.String),
        date="dlycaldt",
        factor=pl.col("dlyprc").abs() / (pl.col("dlyprevprc").abs() * (1 + pl.col("dlyret"))),
    )
    ev = (
        sp.join(links, on="permno", how="left")
        .filter(pl.col("security_id").is_null() | pl.col("date").is_between("vf", "vt"))
        .join(day, on=["permno", "date"], how="left")
    )
    ok = pl.col("security_id").is_not_null() & pl.col("factor").is_finite() & (pl.col("factor") > 0)
    unpriced = ev.filter(~ok.fill_null(False)).select("security_id", "date")
    return ev.filter(ok).select("security_id", "date", "factor", kind=pl.lit("spinoff")), unpriced


def override_events(path: Path, crsp_last: date) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Reviewed spinoff factors after `crsp_last` from adjustment_overrides.csv; rows on
    or before it are obsolete (CRSP's factor is used) and returned separately."""
    schema = {
        "security_id": pl.String,
        "ex_date": pl.String,
        "factor": pl.Float64,
        "note": pl.String,
    }
    df = (
        pl.read_csv(path, schema_overrides=schema) if path.exists() else pl.DataFrame(schema=schema)
    )
    df = df.with_columns(date=pl.col("ex_date").str.to_date())
    live = df.filter(pl.col("date") > crsp_last)
    ev = live.select("security_id", "date", "factor", kind=pl.lit("override"))
    return ev, df.filter(pl.col("date") <= crsp_last).select("security_id", "date", "note")


def combine(*frames: pl.DataFrame) -> pl.DataFrame:
    """All events -> one row per (security_id, date): `factor` (every event) and
    `split_factor` (Massive splits only). On a date with a spinoff or override, that
    security's Massive split and dividend records are dropped: the spinoff factor already
    covers the whole day (Massive logs some spinoffs as splits)."""
    ev = pl.concat([f.select("security_id", "date", "factor", "kind") for f in frames])
    whole_day = (
        ev.filter(pl.col("kind").is_in(["spinoff", "override"]))
        .select("security_id", "date")
        .unique()
    )
    massive_kinds = pl.col("kind").is_in(["split", "dividend"])
    ev = pl.concat(
        [
            ev.filter(massive_kinds).join(whole_day, on=["security_id", "date"], how="anti"),
            ev.filter(~massive_kinds),
        ]
    )
    return (
        ev.group_by("security_id", "date")
        .agg(
            factor=pl.col("factor").product(),
            split_factor=pl.when(pl.col("kind") == "split")
            .then(pl.col("factor"))
            .otherwise(1.0)
            .product(),
        )
        .sort("security_id", "date")
    )


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


# --- Delistings and checks -----------------------------------------------------------------


def delistings(
    prices: pl.DataFrame,
    sm: pl.DataFrame,
    crsp_delist: pl.DataFrame,
    crsp_last: date,
    last_session: date,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One row per security whose bars stop before `last_session`:
    `(ticker, last_bar, delret, source, provisional)`.

    `delret` is CRSP's delisting return of the PERMNO linked on the last bar's date (the
    nearest delisting within DELIST_WINDOW), `source = crsp`. Otherwise it is 0, the last
    traded price as the final value, `source = last_trade`: typically a security that kept
    trading OTC after its exchange delisting (its OTC bars are the realized path, past
    CRSP's record of the PERMNO), or one whose last bar is after `crsp_last`. Rows after
    `crsp_last` are `provisional`: CRSP's next update may supply a return. Also returns the
    `last_trade` rows up to `crsp_last` with the reason CRSP had none, for review."""
    last = prices.group_by("ticker").agg(last_bar=pl.col("timestamp").max())
    stopped = last.filter(pl.col("last_bar").dt.date() < last_session).with_columns(
        day=pl.col("last_bar").dt.date()
    )
    linked = security_master.attach(stopped, sm, "permno", key="ticker", date="day")
    cd = crsp_delist.select(
        permno=pl.col("permno").cast(pl.String), dd="delistingdt", delret="delret"
    )
    lo, hi = DELIST_WINDOW
    near = (
        linked.join(cd, on="permno")
        .filter(pl.col("dd").is_between(pl.col("day") + lo, pl.col("day") + hi))
        .with_columns(gap=(pl.col("dd") - pl.col("day")).dt.total_days().abs())
        .sort("ticker", "gap")
        .unique("ticker", keep="first")
        .select("ticker", "dd", "delret")
    )
    rows = linked.join(near, on="ticker", how="left").with_columns(
        crsp=pl.col("delret").is_not_null() & (pl.col("day") <= crsp_last)
    )
    out = rows.select(
        "ticker",
        "last_bar",
        delret=pl.when("crsp").then("delret").otherwise(0.0),
        source=pl.when("crsp").then(pl.lit("crsp")).otherwise(pl.lit("last_trade")),
        provisional=pl.col("day") > crsp_last,
    )
    review = rows.filter(~pl.col("crsp") & (pl.col("day") <= crsp_last)).select(
        "ticker",
        "last_bar",
        reason=pl.when(pl.col("permno").is_null())
        .then(pl.lit("no PERMNO link on the last bar (e.g. traded OTC after delisting)"))
        .when(pl.col("dd").is_null())
        .then(pl.lit("no CRSP delisting near the last bar"))
        .otherwise(pl.lit("CRSP delisting return missing")),
    )
    return out.sort("ticker"), review.sort("ticker")


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
            daily.select(
                permno=pl.col("permno").cast(pl.String), day="dlycaldt", crsp_ret="dlyret"
            ),
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


def jumps(raw: pl.DataFrame, events: pl.DataFrame, crsp_last: date) -> pl.DataFrame:
    """Unadjusted close-to-close moves beyond JUMP between a security's consecutive bars,
    on days with no adjustment event: candidates for a missed spinoff or a wrong symbol."""
    r = (
        raw.sort("security_id", "date")
        .with_columns(
            move=pl.col("close") / pl.col("close").shift(1).over("security_id") - 1,
            prev_symbol=pl.col("symbol").shift(1).over("security_id"),
        )
        .filter(pl.col("move").abs() > JUMP)
        .join(events.select("security_id", "date"), on=["security_id", "date"], how="anti")
    )
    return r.select(
        "security_id",
        "date",
        "symbol",
        "prev_symbol",
        "move",
        after_crsp=pl.col("date") > crsp_last,
    )


def bad_bars(raw: pl.DataFrame) -> pl.Expr:
    lo, hi = pl.min_horizontal("open", "close"), pl.max_horizontal("open", "close")
    return ~(
        (pl.col("low") > 0)
        & (pl.col("low") <= lo)
        & (hi <= pl.col("high"))
        & (pl.col("volume") >= 0)
    )


# --- The build ----------------------------------------------------------------------------


def build(universe: str, spec: dict, start: date, end: date, root: Path, config_dir: Path) -> Built:
    """Assemble a Datastream membership universe; raises listing every fatal problem
    (security-master or membership invariants, members without a Massive symbol)."""
    ex = {n: wrds.load(root, n)[0] for n in EXTRACTS}
    data = fetch_massive(ex, spec["list_code"], start, end, config_dir)
    return assemble(universe, spec, start, end, ex, data, config_dir)


def fetch_massive(
    ex: dict[str, pl.DataFrame], code: int, start: date, end: date, config_dir: Path
) -> MassiveData:
    """Every Massive pull the build needs: bars for each candidate spelling of each
    security's Datastream tickers (and of override symbols), then those symbols' splits
    and dividends."""
    crsp_last = ex["crsp_daily"]["dlycaldt"].max()
    assert isinstance(crsp_last, date)
    sm = security_master.derive(
        ex["ds_names"], ex["ds_tickers"], ex["ds_cusips"], ex["crsp_security"], crsp_last
    )
    ids = _securities(ex["ds_constituents"], code, start, end)
    tw = ticker_windows(sm, ex["ds_names"], ids, start, end)
    windows = fetch_windows(tw)
    overrides = security_master.read_overrides(config_dir / "universes" / "security_overrides.csv")
    extra = overrides.filter(pl.col("id_type") == "ticker_massive").select(
        symbol="value",
        start=pl.col("valid_from") - timedelta(FETCH_PAD_DAYS),
        end=pl.col("valid_to").fill_null(end) + timedelta(FETCH_PAD_DAYS),
    )
    windows = (
        pl.concat([windows, extra])
        .group_by("symbol")
        .agg(pl.col("start").min(), pl.col("end").max())
    )
    log.info("fetching Massive bars for %d symbols", windows.height)
    bars = fetch_bars(windows)
    used = bars["symbol"].unique().sort().to_list()
    log.info("fetching splits and dividends for %d symbols", len(used))
    splits, divs = fetch_events(used)
    return MassiveData(bars, splits, divs)


def _securities(constituents: pl.DataFrame, code: int, start: date, end: date) -> list[str]:
    """Every security on any list from the last list on or before `start` to `end`."""
    members = membership.list_members(constituents, code, start, end)
    return sorted(members["security_id"].unique())


def assemble(
    universe: str,
    spec: dict,
    start: date,
    end: date,
    ex: dict[str, pl.DataFrame],
    data: MassiveData,
    config_dir: Path,
) -> Built:
    summary: list[str] = []
    reports: dict[str, pl.DataFrame] = {}
    crsp_last = ex["crsp_daily"]["dlycaldt"].max()
    assert isinstance(crsp_last, date)
    sm = security_master.derive(
        ex["ds_names"], ex["ds_tickers"], ex["ds_cusips"], ex["crsp_security"], crsp_last
    )
    ids = _securities(ex["ds_constituents"], spec["list_code"], start, end)
    bars = data.bars.filter(pl.col("date").is_between(start - timedelta(FETCH_PAD_DAYS), end))
    sessions = bars.filter(pl.col("date").is_between(start, end))["date"].unique().sort()

    # Symbols: Datastream ticker intervals -> Massive symbols -> one symbol per session.
    tw = ticker_windows(sm, ex["ds_names"], ids, start, end)
    chosen = choose_symbols(tw, bars)
    reports["ticker_intervals_without_bars"] = chosen.filter(pl.col("symbol").is_null()).select(
        "security_id", "ticker", "vf", "vt"
    )
    assigned = assign_symbols(chosen, bars, sessions)
    sm = pl.concat([sm, runs(assigned)])
    overrides = security_master.read_overrides(config_dir / "universes" / "security_overrides.csv")
    sm = security_master.apply_overrides(sm, overrides).sort("security_id", "id_type", "valid_from")
    if problems := security_master.violations(sm.filter(pl.col("security_id").is_in(ids))):
        raise ValueError(_listing("security master invariants broken", problems))

    used = sessions_for(sm.filter(pl.col("security_id").is_in(ids)), sessions)
    raw_all = used.join(bars, on=["symbol", "date"])  # one bar per (security, session)
    bad = raw_all.filter(bad_bars(raw_all))
    reports["invalid_bars_dropped"] = bad.select("security_id", "date", "symbol", *BAR_COLS)
    raw = raw_all.filter(~bad_bars(raw_all))

    # Membership: per-list members, one class per company, intervals.
    code = spec["list_code"]
    per_list = membership.list_members(ex["ds_constituents"], code, start, end)
    all_lists = membership.list_members(ex["ds_constituents"], code, date(1900, 1, 1), FAR)[
        "list_date"
    ]
    # A list entry Datastream gives no identity (no ticker, no CUSIP, on any date) is a
    # placeholder line (e.g. "APARTMENT INV. & MAN. PLACEHOLDER"), not a security.
    known = sm.filter(pl.col("id_type").is_in(["ticker_datastream", "cusip"]))["security_id"]
    placeholders = per_list.filter(~pl.col("security_id").is_in(known.unique().implode()))
    reports["placeholder_list_entries"] = placeholders.group_by("security_id").agg(
        lists=pl.len(), first=pl.col("list_date").min(), last=pl.col("list_date").max()
    )
    per_list = per_list.join(placeholders, on=["list_date", "security_id"], how="anti")
    if spec.get("one_class_per_company", True):
        liq = raw.select("date", "security_id", dollar_volume=pl.col("close") * pl.col("volume"))
        company = sm.filter(pl.col("id_type") == "company")
        per_list, dropped = membership.keep_one_class(
            per_list, company, liq, spec.get("liquidity_window_bars", 126)
        )
        reports["share_classes_dropped"] = dropped
        twice = membership.company_duplicates(per_list, company)
        if twice.height:
            raise ValueError(
                _listing(
                    "companies with two members on a list",
                    [f"company {r['co']} on {r['list_date']}" for r in twice.iter_rows(named=True)],
                )
            )
    counts = per_list.group_by("list_date").len()
    iv = membership.intervals(per_list, all_lists)
    if problems := membership.violations(iv):
        raise ValueError(_listing("membership invariants broken", problems))

    # Every member session needs a Massive symbol (fatal); a bar is reported if missing.
    member_days = membership.member_rows(
        iv, sessions.to_frame().select(membership.closes(pl.col("date")))["date"]
    )
    md = member_days.with_columns(date=pl.col("timestamp").dt.date())
    # A member that stops trading stays a member until the next list. Only sessions while
    # Datastream still gives it a ticker need a symbol: its ticker history ends (at the
    # latest at its Datastream delisting date) when it stops trading.
    ticker_end = (
        sm.filter(pl.col("id_type") == "ticker_datastream")
        .group_by(ticker="security_id")
        .agg(
            end=pl.when(pl.col("valid_to").is_null().any())
            .then(None)
            .otherwise(pl.col("valid_to").max())
        )
    )
    no_symbol = (
        md.join(used.rename({"security_id": "ticker"}), on=["ticker", "date"], how="anti")
        .join(ticker_end, on="ticker", how="left")
        .filter(pl.col("end").is_null() | (pl.col("date") <= pl.col("end")))
    )
    # Runs of up to MAX_SYMBOL_GAP sessions (a hole between two Datastream ticker rows, as
    # around a seizure or a ticker change) are reported; longer runs are fatal.
    pos = pl.DataFrame({"date": sessions}).with_columns(i=pl.int_range(pl.len()))
    gap_runs = (
        no_symbol.join(pos, on="date")
        .sort("ticker", "i")
        .with_columns(run=(pl.col("i").diff().fill_null(1) != 1).cum_sum().over("ticker"))
        .group_by("ticker", "run")
        .agg(first=pl.col("date").min(), last=pl.col("date").max(), sessions=pl.len())
        .drop("run")
        .sort("ticker", "first")
    )
    reports["member_sessions_without_symbol"] = gap_runs
    gaps = gap_runs.filter(pl.col("sessions") > MAX_SYMBOL_GAP)
    if gaps.height:
        raise ValueError(
            _listing(
                "member sessions without a Massive symbol (add ticker_massive rows to "
                "config/universes/security_overrides.csv)",
                [
                    f"security {r['ticker']}: {r['sessions']} sessions {r['first']}..{r['last']}"
                    for r in gaps.iter_rows(named=True)
                ],
            ),
        )
    # Holes before a security's last bar (after it, a member simply has stopped trading).
    last_bar = raw.group_by(ticker="security_id").agg(last=pl.col("date").max())
    trading = md.join(last_bar, on="ticker").filter(pl.col("date") <= pl.col("last")).drop("last")
    no_bar = trading.join(
        raw.select(ticker="security_id", date="date"), on=["ticker", "date"], how="anti"
    )
    reports["member_sessions_without_bar"] = (
        no_bar.group_by("ticker")
        .agg(sessions=pl.len(), first=pl.col("date").min(), last=pl.col("date").max())
        .sort("ticker")
    )
    permno_gaps = security_master.attach(
        trading.filter(pl.col("date") <= crsp_last), sm, "permno", key="ticker", date="date"
    ).filter(pl.col("permno").is_null())
    reports["member_sessions_without_permno"] = (
        permno_gaps.group_by("ticker")
        .agg(sessions=pl.len(), first=pl.col("date").min(), last=pl.col("date").max())
        .sort("ticker")
    )

    # Adjustment events.
    splits, odd_splits = split_events(data.splits, used)
    divs, skipped_divs = dividend_events(data.dividends, used, raw)
    spins, unpriced = spinoff_events(ex["crsp_distributions"], ex["crsp_daily"], sm, crsp_last)
    spins = spins.filter(pl.col("security_id").is_in(ids))
    manual, obsolete = override_events(
        config_dir / "universes" / "adjustment_overrides.csv", crsp_last
    )
    events = combine(splits, divs, spins, manual)
    reports |= {
        "splits_with_odd_ratio": odd_splits,
        "dividends_skipped": skipped_divs,
        "spinoffs_unpriced": unpriced.filter(pl.col("security_id").is_in(ids)),
        "adjustment_overrides_obsolete": obsolete,
        "unexplained_jumps": jumps(raw, events, crsp_last),
    }

    prices = adjust(raw, events)
    tickers = sorted(prices["ticker"].unique())
    iv = iv.filter(pl.col("security_id").is_in(tickers))
    store.validate(prices, tickers, start, end)

    last_session = sessions.max()
    assert isinstance(last_session, date)
    dl, dl_missing = delistings(prices, sm, ex["crsp_delistings"], crsp_last, last_session)
    reports["delisted_at_last_trade"] = dl_missing
    shares, mismatches = crsp_match(prices, member_days, sm, ex["crsp_daily"], crsp_last)
    reports["crsp_return_mismatches"] = mismatches

    sm_out = sm.filter(pl.col("security_id").is_in(tickers))
    list_dates = per_list["list_date"].unique().sort()
    summary += [
        f"{len(tickers)} securities, {prices.height:,} bars, {iv.height} membership intervals "
        f"over {len(list_dates)} lists ({list_dates.min()} .. {list_dates.max()}), "
        f"{counts['len'].min()}-{counts['len'].max()} members per list",
        f"events: {splits.height} splits, {divs.height} dividends, {spins.height} CRSP spinoffs, "
        f"{manual.height} override factors",
        f"delistings: {dl.filter(pl.col('source') == 'crsp').height} CRSP returns, "
        f"{dl.filter(pl.col('source') == 'last_trade').height} at the last trade "
        f"({dl.filter(pl.col('provisional')).height} provisional, after CRSP's last date)",
        "CRSP match on member sessions: "
        + ", ".join(f"{k} {v:.4f}" for k, v in shares.items() if k != "compared")
        + f" of {int(shares['compared']):,}",
    ]
    summary += [f"report {k}: {v.height} rows" for k, v in reports.items() if v.height]
    manifests = {
        "prices": {
            "source": "massive unadjusted bars x event factors (massive splits/dividends, "
            "crsp spinoffs)",
            "crsp_last": str(crsp_last),
            "tickers": tickers,
        },
        "membership": {
            "source": "datastream",
            "universe": universe,
            "list_code": code,
            "one_class_per_company": bool(spec.get("one_class_per_company", True)),
            "first_list": str(list_dates.min()),
            "last_list": str(list_dates.max()),
            "list_dates": len(list_dates),
        },
        "security_master": {
            "source": "datastream, crsp, massive",
            "securities": len(tickers),
            "id_types": sorted(sm_out["id_type"].unique()),
            "crsp_last": str(crsp_last),
        },
        "delistings": {
            "source": "crsp stkdelists; provisional after crsp_last",
            "crsp_last": str(crsp_last),
        },
    }
    return Built(prices, iv, sm_out, dl, manifests, reports, summary)


def _listing(title: str, problems: list[str], limit: int = 40) -> str:
    more = f"\n  ... and {len(problems) - limit} more" if len(problems) > limit else ""
    return f"{title} ({len(problems)}):\n  " + "\n  ".join(problems[:limit]) + more
