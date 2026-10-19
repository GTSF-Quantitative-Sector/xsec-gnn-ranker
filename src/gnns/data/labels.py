"""Labels: forward returns, entered at the next bar's open and held `horizon_bars` bars.

Labels use data after `t` by definition. They are never builder inputs; only training and
evaluation join them to snapshots on (timestamp, ticker).
"""

import polars as pl


def forward_returns(prices: pl.DataFrame, horizon_bars: int) -> pl.DataFrame:
    """(timestamp, ticker, fwd_return, label_end) for every bar with a defined label.

    fwd_return at axis position i is open[i+1+h] / open[i+1] - 1; label_end is the
    timestamp of bar i+1+h, when the label is fully known. Missing entry/exit bars give no row.
    """
    axis = prices.select(pl.col("timestamp").unique().sort())
    entry, exit_ = pl.col("open").shift(-1), pl.col("open").shift(-(1 + horizon_bars))
    return (
        axis.join(prices.select(pl.col("ticker").unique()), how="cross")
        .join(prices.select("timestamp", "ticker", "open"), on=["timestamp", "ticker"], how="left")
        .sort("ticker", "timestamp")
        .with_columns(
            fwd_return=(exit_ / entry - 1).over("ticker"),
            label_end=pl.col("timestamp").shift(-(1 + horizon_bars)).over("ticker"),
        )
        .drop_nulls(["fwd_return", "label_end"])
        .select("timestamp", "ticker", "fwd_return", "label_end")
        .sort("timestamp", "ticker")
    )
