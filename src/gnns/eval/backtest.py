"""BacktestEngine: a long-only top-k portfolio simulated from a predictions table.

Never sees models or graphs. One rebalance per prediction date; the return until the
next rebalance is one label (open-to-open over `horizon_bars`), so holdings are back to
back. Weights, trades and returns are fractions of the portfolio, so no capital amount is
needed.
"""

import math

import numpy as np
import polars as pl

KEY = ["timestamp", "ticker"]
TRADING_DAYS = 252  # per year, as in metrics.bars_per_year


def target_weights(pred: pl.DataFrame, top_k: int, weighting: str) -> pl.DataFrame:
    """(timestamp, ticker, weight): equal weight in the `top_k` highest scores per date,
    or in every scored name when fewer exist."""
    if weighting != "equal":
        raise ValueError(f"backtest.weighting {weighting!r}: only 'equal' is implemented")
    rank = pl.col("score").rank("ordinal", descending=True).over("timestamp")
    return (
        pred.filter(rank <= top_k)
        .with_columns(weight=1.0 / pl.len().over("timestamp"))
        .select(*KEY, "weight")
    )


def simulate(pred: pl.DataFrame, top_k: int, weighting: str) -> pl.DataFrame:
    """(timestamp, gross, traded) per rebalance: the gross return until the next one, and
    the traded value sum |target - drifted| at it (the first rebalance buys from cash: 1.0)."""
    held = target_weights(pred, top_k, weighting).join(
        pred.select(*KEY, "fwd_return"), on=KEY, how="left"
    )
    _check_held(held)
    gross = (
        held.group_by("timestamp")
        .agg(gross=(pl.col("weight") * pl.col("fwd_return")).sum())
        .sort("timestamp")
    )
    dates = gross["timestamp"]
    next_date = pl.DataFrame({"timestamp": dates[:-1], "next": dates[1:]})
    drifted = (
        held.join(gross, on="timestamp")
        .join(next_date, on="timestamp")  # weights drift into the next rebalance
        .select(
            pl.col("next").alias("timestamp"),
            "ticker",
            drifted=pl.col("weight") * (1 + pl.col("fwd_return")) / (1 + pl.col("gross")),
        )
    )
    diff = pl.col("weight").fill_null(0) - pl.col("drifted").fill_null(0)
    traded = (
        held.select(*KEY, "weight")
        .join(drifted, on=KEY, how="full", coalesce=True)
        .group_by("timestamp")
        .agg(traded=diff.abs().sum())
    )
    return gross.join(traded, on="timestamp").sort("timestamp")


def _check_held(held: pl.DataFrame) -> None:
    """A held name's return must come from the labels; it is never filled or skipped here."""
    bad = held.filter(pl.col("fwd_return").is_null() | pl.col("fwd_return").is_nan())
    if bad.height:
        rows = "\n".join(f"  {t:%Y-%m-%d}  {k}" for t, k in bad.select(KEY).iter_rows())
        raise ValueError(
            f"{bad.height} held positions have no fwd_return:\n{rows}\n"
            "Fix the labels upstream (delisted names: GitHub issue #3)."
        )


def net(bps: float) -> pl.Expr:
    """Return per rebalance after paying `bps` per side on the traded value."""
    return pl.col("gross") - bps * 1e-4 * pl.col("traded")


def drawdown(r: np.ndarray) -> np.ndarray:
    """Fall of the equity curve below its running peak at each rebalance (start value 1)."""
    eq = np.cumprod(1 + r)
    return eq / np.maximum.accumulate(np.maximum(eq, 1.0)) - 1


def summarize(rebalances: pl.DataFrame, cost_bps: list[float], per_year: float) -> dict[str, float]:
    """Metrics for one model. `rebalances` is `simulate` joined with each rebalance's
    `benchmark` and `risk_free` returns; `per_year` is rebalances per year. Cost-dependent
    keys end in `_<bps>bps`."""
    traded = float(rebalances["traded"].to_numpy().mean())
    bench, rf = rebalances["benchmark"].to_numpy(), rebalances["risk_free"].to_numpy()
    active = float((rebalances["gross"].to_numpy() - bench).mean())
    out = {
        "turnover": traded / 2,  # one-way, per rebalance
        "turnover_annual": traded / 2 * per_year,
        # cost at which the mean net return equals the benchmark's
        "break_even_bps": active / traded * 1e4 if traded else math.nan,
    }
    for bps in cost_bps:
        r = rebalances.select(net(bps)).to_series().to_numpy()
        level = _level(r, bench, rf, per_year) | {"cost_drag": bps * 1e-4 * traded * per_year}
        out |= {f"{k}_{bps:g}bps": v for k, v in level.items()}
    return out


def _level(r: np.ndarray, bench: np.ndarray, rf: np.ndarray, per_year: float) -> dict:
    excess, active = r - rf, r - bench
    dd = drawdown(r)
    return {
        "ann_return": float(np.prod(1 + r) ** (per_year / len(r)) - 1),
        "ann_vol": float(r.std(ddof=1) * math.sqrt(per_year)),
        "sharpe": _ratio(excess, per_year),
        "max_drawdown": float(dd.min()),
        # longest stretch below a previous peak, in trading days (comparable across horizons)
        "max_dd_days": _longest_run(dd < 0) * TRADING_DAYS / per_year,
        "excess_return": float(active.mean() * per_year),
        "tracking_error": float(active.std(ddof=1) * math.sqrt(per_year)),
        "ir": _ratio(active, per_year),
        "win_rate": float((excess > 0).mean()),
        "win_rate_vs_benchmark": float((active > 0).mean()),
    }


def _ratio(x: np.ndarray, per_year: float) -> float:
    """Annualized mean / std: Sharpe for excess over cash, IR for excess over benchmark."""
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / sd * math.sqrt(per_year)) if sd > 0 else math.nan


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def yearly(rebalances: pl.DataFrame, bps: float, per_year: float) -> pl.DataFrame:
    """(year, rebalances, net_return, sharpe) per calendar year of the rebalance date."""
    excess = pl.col("r") - pl.col("risk_free")
    return (
        rebalances.select(pl.col("timestamp").dt.year().alias("year"), "risk_free", r=net(bps))
        .group_by("year")
        .agg(
            rebalances=pl.len(),
            net_return=(1 + pl.col("r")).product() - 1,
            sharpe=excess.mean() / excess.std() * math.sqrt(per_year),
        )
        .sort("year")
    )
