"""Adjust, validate, write and load price datasets. Schema: docs/CONTRACTS.md."""

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import polars as pl

PRICE_COLS = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
OHLC = ["open", "high", "low", "close"]


def adjust(prices: pl.DataFrame, dividends: pl.DataFrame) -> pl.DataFrame:
    """Multiply OHLC by the factor of the first dividend with ex-date after the bar's date."""
    # ex_date - 1 day >= bar date  <=>  ex_date > bar date
    divs = dividends.select(
        "ticker", pl.col("ex_date").dt.offset_by("-1d").alias("date"), "factor"
    ).sort("date")
    return (
        prices.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .sort("date")
        .join_asof(divs, on="date", by="ticker", strategy="forward", check_sortedness=False)
        .with_columns(pl.col(c) * pl.col("factor").fill_null(1.0) for c in OHLC)
        .select(PRICE_COLS)
        .sort("timestamp", "ticker")
    )


def validate(prices: pl.DataFrame, tickers: list[str], start: date, end: date) -> None:
    """Raise listing every broken invariant; return silently if the frame is clean."""
    lo, hi = pl.min_horizontal("open", "close"), pl.max_horizontal("open", "close")
    row_checks = {
        "null values": pl.any_horizontal(pl.all().is_null()),
        "duplicate (timestamp, ticker)": pl.struct("timestamp", "ticker").is_duplicated(),
        "bad OHLC (need 0 < low <= open, close <= high)": ~(
            (pl.col("low") > 0) & (pl.col("low") <= lo) & (hi <= pl.col("high"))
        ),
        "volume < 0": pl.col("volume") < 0,
        "ticker not in universe": ~pl.col("ticker").is_in(tickers),
        "timestamp outside [start, end]": ~pl.col("timestamp").dt.date().is_between(start, end),
    }
    counts = prices.select(
        e.fill_null(True).sum().alias(name) for name, e in row_checks.items()
    ).row(0, named=True)
    errors = [f"{name}: {n} rows" for name, n in counts.items() if n]
    if missing := sorted(set(tickers) - set(prices["ticker"])):
        errors.append(f"tickers with no bars: {missing}")
    if errors:
        raise ValueError("validation failed:\n  " + "\n  ".join(errors))


def write(prices: pl.DataFrame, dest: Path, manifest: dict) -> None:
    """Write prices.parquet then manifest.json into `dest`, each atomically."""
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "prices.parquet.tmp"
    prices.write_parquet(tmp)
    sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    os.replace(tmp, dest / "prices.parquet")
    tmp = dest / "manifest.json.tmp"
    tmp.write_text(json.dumps({**manifest, "rows": prices.height, "sha256": sha}, indent=2))
    os.replace(tmp, dest / "manifest.json")


def load(
    root: Path, universe: str, bar: str, start: date | None = None, end: date | None = None
) -> pl.LazyFrame:
    """Lazily scan a dataset, optionally filtered to bar dates in [start, end]."""
    lf = pl.scan_parquet(root / universe / bar / "prices.parquet")
    day = pl.col("timestamp").dt.date()
    if start:
        lf = lf.filter(day >= start)
    if end:
        lf = lf.filter(day <= end)
    return lf
