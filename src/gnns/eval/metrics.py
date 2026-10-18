"""Evaluator metrics on a predictions table (timestamp, ticker, score, fwd_return, ...)."""

import math

import polars as pl


def daily_ic(pred: pl.DataFrame) -> pl.DataFrame:
    """Spearman rank IC between score and forward return at each timestamp."""
    return (
        pred.group_by("timestamp")
        .agg(ic=pl.corr("score", "fwd_return", method="spearman"))
        .filter(pl.col("ic").is_finite())
        .sort("timestamp")
    )


def bars_per_year(bar: str) -> float:
    """Bars in a trading year for a bar size: the only calendar-aware step (CONTRACTS.md)."""
    n, unit = int(bar[:-1]), bar[-1]
    if unit == "d":
        return 252 / n
    if unit == "m":
        return 390 / n * 252  # regular session: 390 minutes per day
    raise ValueError(f"unknown bar size {bar!r}; expected <int>d or <int>m")


def ic_summary(ic: pl.DataFrame, periods_per_year: float = math.nan) -> dict[str, float]:
    """Mean IC, its t-stat, annualized ICIR, hit rate, number of dates.

    icir_annual = mean / std of per-date IC, times sqrt(rebalances per year): comparable
    across horizons. ic_t = icir_annual * sqrt(years of dates), its significance.
    """
    v = ic["ic"].to_numpy()
    if len(v) < 2:
        nan = math.nan
        return {"mean_ic": nan, "ic_t": nan, "icir_annual": nan, "hit_rate": nan, "n_dates": len(v)}
    sd = v.std(ddof=1)
    icir = v.mean() / sd if sd > 0 else math.nan
    return {
        "mean_ic": float(v.mean()),
        "ic_t": float(icir * math.sqrt(len(v))),
        "icir_annual": float(icir * math.sqrt(periods_per_year)),
        "hit_rate": float((v > 0).mean()),
        "n_dates": len(v),
    }


def _rank_desc() -> pl.Expr:
    return pl.col("score").rank("ordinal", descending=True).over("timestamp")


def quantile_returns(pred: pl.DataFrame, quantiles: int) -> pl.DataFrame:
    """(timestamp, quantile, ret): mean forward return per score quantile; 1 = top scores."""
    n = pl.len().over("timestamp")
    return (
        pred.with_columns(quantile=((_rank_desc() - 1) * quantiles // n + 1).cast(pl.Int64))
        .group_by("timestamp", "quantile")
        .agg(ret=pl.col("fwd_return").mean())
        .sort("timestamp", "quantile")
    )


def top_k_returns(pred: pl.DataFrame, k: int) -> pl.DataFrame:
    """(timestamp, ret): equal-weight mean forward return of the k highest scores."""
    return (
        pred.filter(_rank_desc() <= k)
        .group_by("timestamp")
        .agg(ret=pl.col("fwd_return").mean())
        .sort("timestamp")
    )


def universe_returns(pred: pl.DataFrame) -> pl.DataFrame:
    """(timestamp, ret): equal-weight mean forward return of every scored node."""
    return pred.group_by("timestamp").agg(ret=pl.col("fwd_return").mean()).sort("timestamp")
