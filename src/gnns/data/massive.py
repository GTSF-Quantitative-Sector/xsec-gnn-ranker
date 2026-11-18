"""Massive REST API. The only module in gnns.data that touches the network."""

import os
import time as time_mod
from datetime import date, time

import polars as pl
import requests
from dotenv import load_dotenv

from gnns.config import ROOT

BASE_URL = "https://api.massive.com"
TZ = "America/New_York"
RETRY_WAITS = (1, 2, 4, 8, 16)  # seconds before each retry of a transient failure


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
        resp = _request(url, params, headers)
        body = resp.json()
        results.extend(body.get("results", []))
        url, params = body.get("next_url"), {}  # next_url already carries the query
    return results


def _request(url: str, params: dict, headers: dict) -> requests.Response:
    """GET with retries: a rate limit (429), a server error (5xx) or a dropped connection
    is retried after 1, 2, 4, 8, 16 s; anything else raises at once."""
    for wait in (*RETRY_WAITS, None):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code != 429 and resp.status_code < 500:
                resp.raise_for_status()
                return resp
            if wait is None:
                resp.raise_for_status()
        except (requests.ConnectionError, requests.Timeout):
            if wait is None:
                raise
        assert wait is not None
        time_mod.sleep(wait)
    raise AssertionError("unreachable")


def daily_bars(ticker: str, start: date, end: date, adjusted: bool = True) -> pl.DataFrame:
    """Daily bars, split-adjusted unless `adjusted=False`, stamped 16:00 ET (bar end, per
    CONTRACTS.md)."""
    rows = _get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
        {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000},
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


def splits(ticker: str) -> pl.DataFrame:
    """All of a ticker's split records: (ticker, date, split_from, split_to). `date` is the
    execution date, the first bar on the new share basis."""
    rows = _get("/stocks/v1/splits", {"ticker": ticker, "limit": 5000})
    schema = {"execution_date": pl.String, "split_from": pl.Float64, "split_to": pl.Float64}
    return pl.DataFrame(rows, schema=schema).select(
        pl.lit(ticker).alias("ticker"),
        pl.col("execution_date").str.to_date().alias("date"),
        "split_from",
        "split_to",
    )


def cash_dividends(ticker: str) -> pl.DataFrame:
    """All of a ticker's dividend records with their cash amounts:
    (ticker, date, cash_amount, currency, distribution_type); `date` is the ex-date."""
    rows = _get("/stocks/v1/dividends", {"ticker": ticker, "limit": 5000})
    schema = {
        "ex_dividend_date": pl.String,
        "cash_amount": pl.Float64,
        "currency": pl.String,
        "distribution_type": pl.String,
    }
    return pl.DataFrame(rows, schema=schema).select(
        pl.lit(ticker).alias("ticker"),
        pl.col("ex_dividend_date").str.to_date().alias("date"),
        "cash_amount",
        "currency",
        "distribution_type",
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
