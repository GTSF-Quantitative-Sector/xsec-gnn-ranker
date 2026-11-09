"""Build and load the known-at macro dataset: one price series per factor, from Massive."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from gnns.data import massive, store

FACTORS = ("rates", "credit", "vix", "oil", "gold")
# Every factor is a dividend-adjusted price (vix: VXX segments joined on returns), so the
# builder takes fractional returns of all five (docs/graphs/macro_sensitivity.md).
UNITS = dict.fromkeys(FACTORS, "price")
AVAILABILITY = (
    "Massive daily closes, stamped at the 16:00 ET bar end; exchange closing prices "
    "are known at the close and not revised"
)
# A joined series' return on a switch day above this is a broken join, not a market move.
MAX_SEAM_RETURN = 0.25


def fetch_macro(spec: dict, start: date, end: date) -> tuple[pl.DataFrame, dict]:
    """Fetch, adjust, join and validate `spec["series"]`; write nothing.

    Returns the events `(timestamp, factor, value)` and the manifest fields beyond the
    ones `gnns build-data` adds (`built_at`, `git_sha`, `start`, `end`).
    """
    series = spec.get("series", {})
    if set(series) != set(FACTORS):
        raise ValueError(f"sources.macro.series must define exactly {list(FACTORS)}")
    frames = []
    for factor in FACTORS:
        segments = series[factor].get("stitch") or [series[factor]]
        frames.append(_joined(factor, segments, start, end).with_columns(factor=pl.lit(factor)))
    events = _check_events(pl.concat(frames).select("timestamp", "factor", "value"))
    coverage = events.group_by("factor").agg(
        pl.col("timestamp").min().dt.date().cast(pl.String).alias("first"),
        pl.col("timestamp").max().dt.date().cast(pl.String).alias("last"),
    )
    extras = {
        "source": "massive daily bars: " + ", ".join(f"{f} {_tickers(series[f])}" for f in FACTORS),
        "availability_method": AVAILABILITY,
        "known_at_verified": True,
        "units": UNITS,
        "series": json.loads(json.dumps(series, default=str)),
        "coverage": {r["factor"]: [r["first"], r["last"]] for r in coverage.iter_rows(named=True)},
    }
    return events, extras


def load_macro(root: Path, spec: dict) -> tuple[pl.DataFrame, dict]:
    """timestamp means availability, not observation date; no inferred publication lags.

    `spec["path"]` is a dataset folder under `root` (docs/CONTRACTS.md, "Storage layout"),
    written by `gnns build-data` (`fetch_macro`). Each event updates the latest known
    state of one factor. Returns the events and the manifest they were opened with.
    """
    lf, meta = store.open(root / spec["path"])
    if meta.get("known_at_verified") is not True:
        raise ValueError(
            "macro source requires known-at timestamps (known_at_verified: true); "
            "rebuild it with gnns build-data"
        )
    if meta.get("units") != UNITS or not meta.get("availability_method") or not meta["source"]:
        raise ValueError(
            "macro manifest needs source, availability_method and the documented factor "
            "units; rebuild it with gnns build-data"
        )
    return _check_events(lf.collect()), meta


def _check_events(df: pl.DataFrame) -> pl.DataFrame:
    """Raise unless `df` is a valid event table; return it in America/New_York, sorted."""
    if set(df.columns) != {"timestamp", "factor", "value"}:
        raise ValueError("macro source must contain timestamp, factor, value")
    dtype = df.schema["timestamp"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
        raise ValueError("macro timestamps must be tz-aware (an explicit UTC offset)")
    df = df.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.cast_time_unit("us"),
        pl.col("value").cast(pl.Float64),
    )
    if df.is_empty() or df.select(pl.struct("timestamp", "factor").is_duplicated().any()).item():
        raise ValueError("empty macro source or duplicate timestamp/factor")
    values = df["value"].to_numpy()
    if set(df["factor"]) != set(FACTORS) or not (np.isfinite(values) & (values > 0)).all():
        raise ValueError("macro source must contain all five factors with finite positive values")
    return df.sort("timestamp", "factor")


def _tickers(entry: dict) -> str:
    return "/".join(s["ticker"] for s in entry.get("stitch") or [entry])


def _adjusted(ticker: str, start: date, end: date) -> pl.DataFrame:
    """Split- and dividend-adjusted closes `(timestamp, close)`, the way prices are built."""
    bars = store.adjust(massive.daily_bars(ticker, start, end), massive.dividends(ticker))
    return bars.select("timestamp", "close")


def _joined(factor: str, segments: list[dict], start: date, end: date) -> pl.DataFrame:
    """One `(timestamp, value)` series from consecutive tickers, joined on returns.

    Segment i covers bars after segment i-1's `until` through its own `until` (the last
    segment runs to `end`). Values keep the first segment's prices; each later segment is
    rescaled so its first bar's return is its own return from the switch day, or, for a
    renamed note with no bar on that day, the return between the two tickers' unadjusted
    closes.
    """
    if any(s.get("until") is None for s in segments[:-1]):
        raise ValueError(f"{factor}: every stitch segment but the last needs `until`")
    parts: list[pl.DataFrame] = []
    prev: tuple[str, date] | None = None  # previous segment's ticker and until
    for seg in segments:
        until = seg.get("until")
        if until is not None and until >= end:
            until = None  # the build range ends inside this segment: it is the last one
        lo, hi = (prev[1] if prev else start), (until or end)
        bars = _adjusted(seg["ticker"], lo, hi)
        day = pl.col("timestamp").dt.date()
        part = bars.filter(day > prev[1]) if prev else bars
        if part.is_empty():
            raise ValueError(f"{factor}: {seg['ticker']} has no bars in ({lo}, {hi}]")
        if prev is None:
            parts.append(part.rename({"close": "value"}))
        else:
            seam = bars.filter(day == prev[1])
            if not seam.is_empty():
                r = part["close"][0] / seam["close"][0] - 1
            else:  # renamed note: no bar on the switch day in the new ticker
                first = part["timestamp"][0].date()
                old = massive.daily_bars(prev[0], prev[1], prev[1], adjusted=False)["close"]
                new = massive.daily_bars(seg["ticker"], first, first, adjusted=False)["close"]
                if old.is_empty() or new.is_empty():
                    raise ValueError(
                        f"{factor}: no unadjusted bars to join {prev[0]} and {seg['ticker']}"
                    )
                r = new[0] / old[0] - 1
            if abs(r) > MAX_SEAM_RETURN:
                raise ValueError(
                    f"{factor}: return {r:+.1%} joining {prev[0]} -> {seg['ticker']} after "
                    f"{prev[1]}; the join is broken"
                )
            scale = parts[-1]["value"][-1] * (1 + r) / part["close"][0]
            parts.append(part.select("timestamp", (pl.col("close") * scale).alias("value")))
        if until is None:
            break
        if parts[-1]["timestamp"][-1].date() != until:
            raise ValueError(f"{factor}: {seg['ticker']} ends before its `until` {until}")
        prev = (seg["ticker"], until)
    return pl.concat(parts)
