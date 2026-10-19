"""Massive REST API. The only module in gnns.data that touches the network."""

import os
from datetime import date, time

import polars as pl
import requests
from dotenv import load_dotenv

from gnns.config import ROOT

BASE_URL = "https://api.massive.com"
TZ = "America/New_York"


def _get(path: str, params: dict) -> list[dict]:
    """GET every page of an endpoint and return the concatenated `results`."""
    load_dotenv(ROOT / ".env")
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise RuntimeError("MASSIVE_API_KEY is not set; add it to .env (see README).")
    headers = {"Authorization": f"Bearer {key}"}
    url: str | None = BASE_URL + path
    results: list[dict] = []
    while url:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        results.extend(body.get("results", []))
        url, params = body.get("next_url"), {}  # next_url already carries the query
    return results


def daily_bars(ticker: str, start: date, end: date) -> pl.DataFrame:
    """Split-adjusted daily bars, stamped 16:00 ET (bar end, per CONTRACTS.md)."""
    rows = _get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
        {"adjusted": "true", "sort": "asc", "limit": 50000},
    )
    schema = {c: pl.Float64 for c in "ohlcv"} | {"t": pl.Int64}
    return pl.DataFrame(rows, schema=schema).select(
        pl.from_epoch("t", time_unit="ms")  # `t` is the bar start, Unix ms UTC
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(TZ)
        .dt.date()
        .dt.combine(time(16))
        .dt.replace_time_zone(TZ)
        .alias("timestamp"),
        pl.lit(ticker).alias("ticker"),
        pl.col("o").alias("open"),
        pl.col("h").alias("high"),
        pl.col("l").alias("low"),
        pl.col("c").alias("close"),
        pl.col("v").alias("volume"),
    )


def dividends(ticker: str) -> pl.DataFrame:
    """All of a ticker's dividends, unfiltered by date: a bar's factor comes from the first
    ex-date after it, even when that falls outside the build window."""
    rows = _get("/stocks/v1/dividends", {"ticker": ticker, "limit": 5000})
    schema = {"ex_dividend_date": pl.String, "historical_adjustment_factor": pl.Float64}
    return pl.DataFrame(rows, schema=schema).select(
        pl.lit(ticker).alias("ticker"),
        pl.col("ex_dividend_date").str.to_date().alias("ex_date"),
        pl.col("historical_adjustment_factor").alias("factor"),
    )
