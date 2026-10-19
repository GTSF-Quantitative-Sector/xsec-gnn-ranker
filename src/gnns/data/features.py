"""FeatureEngine: shared equity node features, computed over a whole price panel at once.

Windows count positions on the bar axis (docs/CONTRACTS.md), so each ticker is first
spread onto the full axis, with missing bars as nulls.
"""

import logging
from collections.abc import Callable

import polars as pl

log = logging.getLogger(__name__)


def momentum(bars: int) -> pl.Expr:
    """Return over the last `bars` bars."""
    return pl.col("close") / pl.col("close").shift(bars) - 1


def volatility(bars: int) -> pl.Expr:
    """Standard deviation of 1-bar returns over the last `bars` bars."""
    return (pl.col("close") / pl.col("close").shift(1) - 1).rolling_std(bars)


FEATURES: dict[str, Callable[..., pl.Expr]] = {"momentum": momentum, "volatility": volatility}


def column_names(specs: list[dict]) -> list[str]:
    """Output column of each feature entry, e.g. {name: momentum, bars: 20} -> momentum_20."""
    return ["_".join([s["name"], *(str(v) for k, v in s.items() if k != "name")]) for s in specs]


def compute(prices: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """One row per eligible (timestamp, ticker), one column per configured feature."""
    specs, min_history = cfg["list"], cfg["min_history_bars"]
    if unknown := sorted({s["name"] for s in specs} - FEATURES.keys()):
        raise ValueError(f"unknown features {unknown}; valid: {sorted(FEATURES)}")
    if min_history < (longest := max(s["bars"] for s in specs)):
        raise ValueError(f"min_history_bars={min_history} < longest feature window {longest}")
    params = [{k: v for k, v in s.items() if k != "name"} for s in specs]
    names = column_names(specs)

    # Dense bar axis per ticker, from its first bar to its last.
    axis = prices.select(pl.col("timestamp").unique())
    span = prices.group_by("ticker").agg(
        first=pl.col("timestamp").min(), last=pl.col("timestamp").max()
    )
    dense = (
        axis.join(span, how="cross")
        .filter(pl.col("timestamp").is_between("first", "last"))
        .join(prices.select("timestamp", "ticker", "close"), on=["timestamp", "ticker"], how="left")
        .sort("ticker", "timestamp")
    )

    out = (
        dense.with_columns(
            FEATURES[s["name"]](**p).over("ticker").alias(n)
            for s, p, n in zip(specs, params, names, strict=True)
        )
        .with_columns(age=pl.int_range(pl.len()).over("ticker"))
        .filter(pl.col("age") >= min_history)
        .with_columns(pl.when(pl.col(n).is_finite()).then(pl.col(n)) for n in names)
    )

    nulls = out.select(pl.col(names).null_count()).row(0, named=True)
    if any(nulls.values()):
        log.info("median-filling feature values: %s", {n: c for n, c in nulls.items() if c})
    out = out.with_columns(pl.col(n).fill_null(pl.col(n).median().over("timestamp")) for n in names)
    return out.select("timestamp", "ticker", *names).drop_nulls().sort("timestamp", "ticker")
