"""Labels: forward returns, entered at the next bar's open and held `horizon_bars` bars.

Labels use data after `t` by definition. They are never builder inputs; only training and
evaluation join them to snapshots on (timestamp, ticker). Contract: docs/CONTRACTS.md,
"Labels".
"""

from pathlib import Path

import polars as pl

from gnns.data import store


def forward_returns(
    prices: pl.DataFrame, horizon_bars: int, delistings: pl.DataFrame | None = None
) -> pl.DataFrame:
    """(timestamp, ticker, fwd_return, label_end) for every bar with a defined label.

    fwd_return at axis position i is open[i+1+h] / open[i+1] - 1; label_end is the
    timestamp of bar i+1+h, when the label is fully known. Only positions with
    i + 1 + h <= the last position have labels.

    With `delistings` (`ticker, last_bar, delret`), a ticker whose exit bar is missing
    because its last bar L falls inside the window (i <= L < i+1+h) is labeled from its
    final value, close[L] × (1 + delret): fwd_return = close[L] × (1 + delret) / open[i+1]
    - 1, or delret itself when L = i (the signal bar is its last). Any other missing entry
    or exit bar gives no row.
    """
    delisting = delistings is not None and delistings.height > 0
    axis = prices.select(pl.col("timestamp").unique().sort()).with_row_index("pos")
    last_pos = axis.height - 1
    entry, exit_ = pl.col("open").shift(-1), pl.col("open").shift(-(1 + horizon_bars))
    cols = ["timestamp", "ticker", "open", *(["close"] if delisting else [])]
    dense = (
        axis.join(prices.select(pl.col("ticker").unique()), how="cross")
        .join(prices.select(cols), on=["timestamp", "ticker"], how="left")
        .sort("ticker", "timestamp")
        .with_columns(
            entry=entry.over("ticker"),
            fwd_return=(exit_ / entry - 1).over("ticker"),
            label_end=pl.col("timestamp").shift(-(1 + horizon_bars)).over("ticker"),
        )
        .filter(pl.col("pos") + 1 + horizon_bars <= last_pos)
    )
    if delisting:
        assert delistings is not None
        final = (
            delistings.select("ticker", "last_bar", "delret")
            .join(axis.rename({"timestamp": "last_bar", "pos": "last"}), on="last_bar")
            .join(
                prices.select("ticker", last_bar="timestamp", last_close="close"),
                on=["ticker", "last_bar"],
            )
            .select("ticker", "last", value=pl.col("last_close") * (1 + pl.col("delret")))
        )
        dense = dense.join(final, on="ticker", how="left").with_columns(
            fwd_return=pl.when(
                pl.col("fwd_return").is_null()
                & pl.col("last").is_between(pl.col("pos"), pl.col("pos") + horizon_bars)
            )
            .then(
                pl.when(pl.col("last") == pl.col("pos"))
                .then(pl.col("value") / pl.col("close") - 1)
                .otherwise(pl.col("value") / pl.col("entry") - 1)
            )
            .otherwise(pl.col("fwd_return"))
        )
    return (
        dense.drop_nulls(["fwd_return", "label_end"])
        .select("timestamp", "ticker", "fwd_return", "label_end")
        .sort("timestamp", "ticker")
    )


def load_delistings(root: Path, universe: str) -> pl.DataFrame | None:
    """A membership universe's delistings (`ticker, last_bar, delret`), or None for a
    universe without one (list universes)."""
    dest = root / universe / "delistings"
    if not (dest / "manifest.json").exists():
        return None
    lf, _ = store.open(dest)
    return lf.collect()
