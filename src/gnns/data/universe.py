"""Build a membership universe (the central `sp500`): the universe table, the raw bars its
prices are adjusted from, the security master and the delistings.

Contract: docs/CONTRACTS.md ("Membership universes", "Security master"). `gnns
build-universe` calls `build`: `fetch_bars` (Massive, the only network step), then
`assemble`, a pure function of the cached WRDS extracts and the bars (tested on fixtures).
The universe table holds every member on every session it can trade; restrictions
(`gnns.data.restrict`) narrow it to a construction's tradeable set.
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

from gnns.data import massive, membership, restrict, security_master, store, wrds
from gnns.data.security_master import FAR

log = logging.getLogger(__name__)
TZ = "America/New_York"
# Extract -> columns this build reads (an older cache lacks some: re-run gnns fetch-wrds).
EXTRACTS = {
    "ds_constituents": ["indexlistintcode", "infocode", "startdate", "enddate"],
    "ds_tickers": ["infocode", "startdate", "enddate", "ticker"],
    "ds_cusips": ["infocode", "startdate", "enddate", "cusip"],
    "ds_names": ["infocode", "startdate", "enddate", "dscmpycode", "isin", "dsqtname"],
    "crsp_security": ["permno", "permco", "secinfostartdt", "secinfoenddt", "cusip", "ticker"],
    "crsp_delistings": ["permno", "delistingdt", "delret"],
    "crsp_daily": ["permno", "permco", "dlycaldt", "dlyprc", "dlycap", "dlydelflg"],
}
FETCH_PAD_DAYS = 10  # extra days of bars around each symbol's window
BOUNDARY_DAYS = 7  # around a ticker change, the neighbouring symbol may carry the bars
MAX_SEAM = 0.2  # a borrowed symbol's close must be within this |log move| of the last close
MAX_SYMBOL_GAP = 5  # member sessions in a row without a symbol before the build fails
WORKERS = 8
CLASS_MARK = re.compile(r"'[A-Z]'")  # Datastream marks share classes like ALPHABET 'A'
BAR_COLS = ["open", "high", "low", "close", "volume", "vwap"]
ID_COLUMNS = (
    "ticker_massive",
    "ticker_datastream",
    "ticker_crsp",
    "cusip",
    "isin",
    "company",
    "permno",
    "permco",
)
CAP_UNIT = 1000.0  # CRSP's dlycap is in thousands of dollars
DEFAULT_ADV_WINDOWS = (20, 63)


@dataclass
class Built:
    """Everything a membership universe's build produces, unwritten."""

    universe: pl.DataFrame
    raw_bars: pl.DataFrame
    security_master: pl.DataFrame
    delistings: pl.DataFrame
    manifests: dict[str, dict]  # dataset -> manifest fields beyond built_at/git_sha
    reports: dict[str, pl.DataFrame] = field(default_factory=dict)
    summary: list[str] = field(default_factory=list)


def schema(adv_windows: tuple[int, ...] | list[int] = DEFAULT_ADV_WINDOWS) -> dict:
    """The universe table's columns (`universe.parquet`, docs/CONTRACTS.md)."""
    return {
        "timestamp": pl.Datetime("us", TZ),
        "security_id": pl.String,
        "list_date": pl.Date,
        **dict.fromkeys(ID_COLUMNS, pl.String),
        "dollar_volume": pl.Float64,
        **{f"adv_{w}": pl.Float64 for w in adv_windows},
        "mktcap": pl.Float64,
    }


# --- Reading the built table ----------------------------------------------------------------


def exists(root: Path, name: str) -> bool:
    return (root / name / "universe" / "manifest.json").exists()


def load(root: Path, name: str) -> tuple[pl.DataFrame, dict]:
    """A membership universe's table and manifest; raises if it hasn't been built."""
    dest = root / name / "universe"
    if not (dest / "manifest.json").exists():
        raise FileNotFoundError(f"{dest} missing; run gnns build-universe first")
    lf, manifest = store.open(dest)
    return lf.collect(), manifest


def mask(root: Path, name: str, filters: list[dict] | None = None) -> pl.DataFrame:
    """(timestamp, ticker) for every session at which a security is in the universe after
    `filters` (`gnns.data.restrict`); `ticker` holds the security id, the node key."""
    u, _ = load(root, name)
    u = restrict.apply(u, filters or [])
    return u.select("timestamp", ticker="security_id").sort("timestamp", "ticker")


# --- The build ------------------------------------------------------------------------------


def build(universe: str, spec: dict, start: date, end: date, root: Path, config_dir: Path) -> Built:
    """Assemble a Datastream membership universe; raises listing every fatal problem
    (security-master or membership invariants, members without a Massive symbol)."""
    ex = load_extracts(root)
    crsp_last = _crsp_last(ex)
    if end > crsp_last:
        raise ValueError(
            f"data.end {end} is after CRSP's last date {crsp_last}: set data.end in "
            f"config/config.yaml to {crsp_last} or earlier (prices are adjusted with CRSP's "
            "factors, which end there)"
        )
    windows = fetch_windows_for(ex, spec["list_code"], start, end, config_dir)
    log.info("fetching Massive bars for %d symbols", windows.height)
    return assemble(universe, spec, start, end, ex, fetch_bars(windows), config_dir)


def load_extracts(root: Path) -> dict[str, pl.DataFrame]:
    """The cached WRDS extracts this build reads; raises if one predates a needed column."""
    ex = {}
    for name, cols in EXTRACTS.items():
        df, _ = wrds.load(root, name)
        if missing := sorted(set(cols) - set(df.columns)):
            raise ValueError(f"WRDS extract {name} lacks {missing}; re-run gnns fetch-wrds")
        ex[name] = df
    return ex


def _crsp_last(ex: dict[str, pl.DataFrame]) -> date:
    last = ex["crsp_daily"]["dlycaldt"].max()
    assert isinstance(last, date)
    return last


def fetch_windows_for(
    ex: dict[str, pl.DataFrame], code: int, start: date, end: date, config_dir: Path
) -> pl.DataFrame:
    """(symbol, start, end) for every candidate spelling of each security's Datastream
    tickers, and for every override symbol."""
    sm = _derive(ex)
    ids = _securities(ex["ds_constituents"], code, start, end)
    windows = fetch_windows(ticker_windows(sm, ex["ds_names"], ids, start, end))
    overrides = security_master.read_overrides(config_dir / "universes" / "security_overrides.csv")
    extra = overrides.filter(pl.col("id_type") == "ticker_massive").select(
        symbol="value",
        start=pl.col("valid_from") - timedelta(FETCH_PAD_DAYS),
        end=pl.col("valid_to").fill_null(end) + timedelta(FETCH_PAD_DAYS),
    )
    return (
        pl.concat([windows, extra])
        .group_by("symbol")
        .agg(pl.col("start").min(), pl.col("end").max())
        .sort("symbol")
    )


def _derive(ex: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return security_master.derive(
        ex["ds_names"], ex["ds_tickers"], ex["ds_cusips"], ex["crsp_security"], _crsp_last(ex)
    )


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
    bars: pl.DataFrame,
    config_dir: Path,
) -> Built:
    """The universe's datasets from the extracts and Massive's unadjusted `bars`
    (`symbol, date, timestamp, open, high, low, close, volume, vwap`)."""
    reports: dict[str, pl.DataFrame] = {}
    crsp_last = _crsp_last(ex)
    code = spec["list_code"]
    adv_windows = tuple(spec.get("adv_windows", DEFAULT_ADV_WINDOWS))
    sm = _derive(ex)
    ids = _securities(ex["ds_constituents"], code, start, end)
    bars = bars.filter(pl.col("date").is_between(start - timedelta(FETCH_PAD_DAYS), end))
    sessions = bars.filter(pl.col("date").is_between(start, end))["date"].unique().sort()

    # Symbols: Datastream ticker intervals -> Massive symbols -> one symbol per session.
    tw = ticker_windows(sm, ex["ds_names"], ids, start, end)
    chosen = choose_symbols(tw, bars)
    reports["ticker_intervals_without_bars"] = chosen.filter(pl.col("symbol").is_null()).select(
        "security_id", "ticker", "vf", "vt"
    )
    sm = pl.concat([sm, runs(assign_symbols(chosen, bars, sessions))])
    overrides = security_master.read_overrides(config_dir / "universes" / "security_overrides.csv")
    sm = security_master.apply_overrides(sm, overrides).sort("security_id", "id_type", "valid_from")
    if problems := security_master.violations(sm.filter(pl.col("security_id").is_in(ids))):
        raise ValueError(_listing("security master invariants broken", problems))

    # Raw bars: one per (security, session), invalid ones dropped, cut at CRSP's delisting.
    used = sessions_for(sm.filter(pl.col("security_id").is_in(ids)), sessions)
    raw = used.join(bars, on=["symbol", "date"])
    reports["invalid_bars_dropped"] = raw.filter(bad_bars()).select(
        "security_id", "date", "symbol", *BAR_COLS[:5]
    )
    raw = raw.filter(~bad_bars())
    ends, no_return = crsp_ends(sm, ex["crsp_delistings"], ex["crsp_daily"], crsp_last)
    raw, after = cut_at_delisting(raw, ends)
    reports["bars_after_crsp_delisting"] = after
    reports["delistings_without_return"] = no_return

    # Membership: per-list members (placeholders dropped) -> intervals -> member sessions.
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
    iv = membership.intervals(per_list, all_lists)
    if problems := membership.violations(iv):
        raise ValueError(_listing("membership invariants broken", problems))
    members = member_sessions(iv, per_list["list_date"].unique(), sessions)

    gaps = symbol_gaps(members, used, sm, sessions)
    reports["member_sessions_without_symbol"] = gaps
    if (fatal := gaps.filter(pl.col("sessions") > MAX_SYMBOL_GAP)).height:
        raise ValueError(
            _listing(
                "member sessions without a Massive symbol (add ticker_massive rows to "
                "config/universes/security_overrides.csv)",
                [
                    f"security {r['security_id']}: {r['sessions']} sessions "
                    f"{r['first']}..{r['last']}"
                    for r in fatal.iter_rows(named=True)
                ],
            )
        )

    # A member is in the table on every session up to its last bar.
    last_bar = raw.group_by("security_id").agg(last=pl.col("date").max())
    rows = members.join(last_bar, on="security_id").filter(pl.col("date") <= pl.col("last"))
    rows = rows.drop("last")
    reports["member_sessions_without_bar"] = _runs_per_security(
        rows.join(raw.select("security_id", "date"), on=["security_id", "date"], how="anti")
    )
    for id_type in ID_COLUMNS:
        rows = security_master.attach(rows, sm, id_type, key="security_id", date="date")
    reports["member_sessions_without_permno"] = _runs_per_security(
        rows.filter(pl.col("permno").is_null())
    )
    rows = rows.join(liquidity(raw, adv_windows), on=["security_id", "date"], how="left")
    rows = _carry_adv(rows, raw, adv_windows)
    rows = rows.join(market_caps(ex["crsp_daily"]), on=["permco", "date"], how="left")
    reports["member_sessions_without_mktcap"] = _runs_per_security(
        rows.filter(pl.col("mktcap").is_null())
    )
    table = rows.select(list(schema(adv_windows))).sort("timestamp", "security_id")

    # Delistings: the CRSP return of every security cut at its delisting.
    final = raw.sort("date").group_by("security_id").agg(last_bar=pl.col("timestamp").last())
    dl = (
        ends.drop_nulls("delret")
        .join(final, on="security_id")
        .select(ticker="security_id", last_bar="last_bar", delret="delret")
        .sort("ticker")
    )
    last_session = sessions.max()
    stopped = final.filter(pl.col("last_bar").dt.date() < last_session)
    reports["stopped_without_crsp_delisting"] = stopped.join(ends, on="security_id", how="anti")

    raw_out = raw.select("timestamp", "security_id", "symbol", *BAR_COLS).sort(
        "timestamp", "security_id"
    )
    tickers = sorted(raw_out["security_id"].unique())
    sm_out = sm.filter(pl.col("security_id").is_in(tickers))
    list_dates = per_list["list_date"].unique().sort()
    per_day = table.group_by("timestamp").len()["len"]
    summary = [
        f"{len(tickers)} securities, {raw_out.height:,} bars, {table.height:,} member sessions "
        f"over {len(list_dates)} lists ({list_dates.min()} .. {list_dates.max()}), "
        f"{per_day.min()}-{per_day.max()} members per session (all share classes)",
        f"delistings: {dl.height} with a CRSP return",
    ]
    summary += [f"report {k}: {v.height} rows" for k, v in reports.items() if v.height]
    window = {"start": str(start), "end": str(end), "crsp_last": str(crsp_last)}
    manifests = {
        "universe": {
            "source": "datastream lists; crsp, datastream and massive identifiers; massive "
            "dollar volume; crsp market cap",
            "universe": universe,
            "list_code": code,
            "first_list": str(list_dates.min()),
            "last_list": str(list_dates.max()),
            "list_dates": len(list_dates),
            "adv_windows": list(adv_windows),
            "securities": len(tickers),
        }
        | window,
        "raw_bars": {
            "source": "massive unadjusted daily bars",
            "universe": universe,
            "tickers": tickers,
        }
        | window,
        "security_master": {
            "source": "datastream, crsp, massive",
            "securities": len(tickers),
            "id_types": sorted(sm_out["id_type"].unique()),
            "crsp_last": str(crsp_last),
        },
        "delistings": {
            "source": "crsp stkdelists",
            "universe": universe,
            "crsp_last": str(crsp_last),
        },
    }
    return Built(table, raw_out, sm_out, dl, manifests, reports, summary)


# --- Massive --------------------------------------------------------------------------------


def _parallel(fn: Callable, items: list) -> list:
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(fn, items))


def fetch_bars(windows: pl.DataFrame) -> pl.DataFrame:
    """Unadjusted daily bars (with VWAP) for each `(symbol, start, end)` window."""

    def one(row: tuple) -> pl.DataFrame:
        sym, lo, hi = row
        return massive.daily_bars(sym, lo, hi, adjusted=False, vwap=True)

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


def _empty_bars() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("us", TZ),
            "symbol": pl.String,
            **dict.fromkeys(BAR_COLS, pl.Float64),
            "date": pl.Date,
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


def bad_bars() -> pl.Expr:
    """Bars breaking the OHLC invariants (docs/CONTRACTS.md, "prices.parquet")."""
    lo, hi = pl.min_horizontal("open", "close"), pl.max_horizontal("open", "close")
    return ~(
        (pl.col("low") > 0)
        & (pl.col("low") <= lo)
        & (hi <= pl.col("high"))
        & (pl.col("volume") >= 0)
    )


# --- CRSP: delistings and market cap --------------------------------------------------------


def crsp_ends(
    sm: pl.DataFrame, crsp_delist: pl.DataFrame, crsp_daily: pl.DataFrame, crsp_last: date
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Securities whose final PERMNO (the last one linked) CRSP delists by `crsp_last`:
    `(security_id, last_day, delret)`, where `last_day` is that PERMNO's last CRSP trading
    day (its last daily row with a price that is not a delisting record) and `delret`
    CRSP's return from that day's price to the final value (null if CRSP has none).
    Also returns the rows without a return, for review."""
    final = (
        sm.filter(pl.col("id_type") == "permno")
        .sort("security_id", "valid_from")
        .group_by("security_id")
        .agg(permno=pl.col("value").last())
    )
    dl = (
        crsp_delist.filter(pl.col("delistingdt") <= crsp_last)
        .select(permno=_permno("permno"), dd="delistingdt", delret="delret")
        .sort("dd")
        .unique("permno", keep="last")
    )
    traded = (
        crsp_daily.filter(
            pl.col("dlyprc").is_not_null() & (pl.col("dlydelflg").fill_null("N") != "Y")
        )
        .group_by(permno=_permno("permno"))
        .agg(last_day=pl.col("dlycaldt").max())
    )
    ends = (
        final.join(dl, on="permno")
        .join(traded, on="permno", how="left")
        .select("security_id", last_day=pl.col("last_day").fill_null(pl.col("dd")), delret="delret")
        .sort("security_id")
    )
    return ends, ends.filter(pl.col("delret").is_null()).select("security_id", "last_day")


def _permno(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Int64).cast(pl.String)


def cut_at_delisting(raw: pl.DataFrame, ends: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Drop every bar after a delisted security's last CRSP trading day (e.g. OTC trading
    under a Q ticker): its delisting return is the final value. Returns the kept bars and,
    per security, the dropped ones."""
    r = raw.join(ends.select("security_id", "last_day"), on="security_id", how="left")
    after = pl.col("last_day").is_not_null() & (pl.col("date") > pl.col("last_day"))
    dropped = (
        r.filter(after)
        .group_by("security_id")
        .agg(
            bars=pl.len(),
            first=pl.col("date").min(),
            last=pl.col("date").max(),
            symbols=pl.col("symbol").unique().sort().str.join(","),
        )
        .sort("security_id")
    )
    return r.filter(~after).drop("last_day"), dropped


def market_caps(crsp_daily: pl.DataFrame) -> pl.DataFrame:
    """(permco, date, mktcap): each company's market cap in dollars, the sum of `dlycap`
    over every security of the PERMCO CRSP prices that day (each listed share class)."""
    return (
        crsp_daily.filter(
            pl.col("dlycap").is_not_null() & (pl.col("dlydelflg").fill_null("N") != "Y")
        )
        .group_by(permco=_permno("permco"), date="dlycaldt")
        .agg(mktcap=pl.col("dlycap").sum() * CAP_UNIT)
    )


# --- Membership rows, liquidity, coverage ---------------------------------------------------


def member_sessions(iv: pl.DataFrame, list_dates: pl.Series, sessions: pl.Series) -> pl.DataFrame:
    """(timestamp, security_id, date, list_date) for every session at which a security is
    a member; `list_date` is the latest list on or before the session (the one membership
    at that session follows)."""
    ts = sessions.to_frame("date").select(membership.closes(pl.col("date")))["date"]
    lists = pl.DataFrame({"list_date": list_dates.unique().sort()})
    return (
        membership.member_rows(iv, ts)
        .select("timestamp", security_id="ticker", date=pl.col("timestamp").dt.date())
        .sort("date")
        .join_asof(lists, left_on="date", right_on="list_date", strategy="backward")
        .sort("timestamp", "security_id")
    )


def liquidity(raw: pl.DataFrame, windows: tuple[int, ...]) -> pl.DataFrame:
    """(security_id, date, dollar_volume, adv_<w>...) on each bar: `vwap × volume` (`close ×
    volume` where Massive gives no VWAP) and its trailing mean over the last `w` bars up to
    and including the date (fewer at the start of a security's history)."""
    dv = pl.coalesce(pl.col("vwap"), pl.col("close")) * pl.col("volume")
    return (
        raw.sort("security_id", "date")
        .select("security_id", "date", dollar_volume=dv)
        .with_columns(
            pl.col("dollar_volume")
            .rolling_mean(w, min_samples=1)
            .over("security_id")
            .alias(f"adv_{w}")
            for w in windows
        )
    )


def _carry_adv(rows: pl.DataFrame, raw: pl.DataFrame, windows: tuple[int, ...]) -> pl.DataFrame:
    """On a member session without a bar (a halt), each `adv_<w>` is the last bar's."""
    cols = [f"adv_{w}" for w in windows]
    if not rows.filter(pl.col(cols[0]).is_null()).height:
        return rows
    last = (
        liquidity(raw, windows)
        .select("security_id", "date", *(pl.col(c).alias(f"_{c}") for c in cols))
        .sort("date")
    )
    return (
        rows.sort("date")
        .join_asof(last, on="date", by="security_id", strategy="backward", check_sortedness=False)
        .with_columns(pl.coalesce(c, f"_{c}").alias(c) for c in cols)
        .drop(f"_{c}" for c in cols)
    )


def symbol_gaps(
    members: pl.DataFrame, used: pl.DataFrame, sm: pl.DataFrame, sessions: pl.Series
) -> pl.DataFrame:
    """Runs of member sessions without a Massive symbol: `(security_id, first, last,
    sessions)`. A member that stops trading stays a member until the next list, so only
    sessions while Datastream still gives it a ticker count (its ticker history ends, at
    the latest at its Datastream delisting date, when it stops trading)."""
    ticker_end = (
        sm.filter(pl.col("id_type") == "ticker_datastream")
        .group_by("security_id")
        .agg(
            end=pl.when(pl.col("valid_to").is_null().any())
            .then(None)
            .otherwise(pl.col("valid_to").max())
        )
    )
    no_symbol = (
        members.join(used, on=["security_id", "date"], how="anti")
        .join(ticker_end, on="security_id", how="left")
        .filter(pl.col("end").is_null() | (pl.col("date") <= pl.col("end")))
    )
    pos = pl.DataFrame({"date": sessions.unique().sort()}).with_columns(i=pl.int_range(pl.len()))
    return (
        no_symbol.join(pos, on="date")
        .sort("security_id", "i")
        .with_columns(run=(pl.col("i").diff().fill_null(1) != 1).cum_sum().over("security_id"))
        .group_by("security_id", "run")
        .agg(first=pl.col("date").min(), last=pl.col("date").max(), sessions=pl.len())
        .drop("run")
        .sort("security_id", "first")
    )


def _runs_per_security(rows: pl.DataFrame) -> pl.DataFrame:
    """(security_id, sessions, first, last) of the given member sessions."""
    return (
        rows.group_by("security_id")
        .agg(sessions=pl.len(), first=pl.col("date").min(), last=pl.col("date").max())
        .sort("security_id")
    )


def _listing(title: str, problems: list[str], limit: int = 40) -> str:
    more = f"\n  ... and {len(problems) - limit} more" if len(problems) > limit else ""
    return f"{title} ({len(problems)}):\n  " + "\n  ".join(problems[:limit]) + more
