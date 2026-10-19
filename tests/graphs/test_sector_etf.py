from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from gnns.data import features
from gnns.graphs.sector_etf import SectorETFCorrelationBuilder

FEATURES_CFG = {"min_history_bars": 5, "list": [{"name": "momentum", "bars": 5}]}


def _prices(n_bars: int = 20) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    t0 = datetime(2024, 1, 1, 16, tzinfo=ZoneInfo("America/New_York"))
    closes = {k: 100 * np.cumprod(1 + rng.normal(0, 0.02, n_bars)) for k in ("XLE", "XLF", "XLK")}
    return pl.DataFrame(
        [
            (t0 + timedelta(days=i), k, float(c[i]))
            for k, c in closes.items()
            for i in range(n_bars)
        ],
        schema=["timestamp", "ticker", "close"],
        orient="row",
    )


def _builder(prices: pl.DataFrame) -> SectorETFCorrelationBuilder:
    inputs = {"prices": prices, "features": features.compute(prices, FEATURES_CFG)}
    return SectorETFCorrelationBuilder(inputs, prices["timestamp"].unique(), window_bars=5)


def test_fully_connected_weighted_by_abs_correlation():
    prices = _prices()
    t = prices["timestamp"].sort()[-1]
    s = _builder(prices).build(t)

    assert s.node_ids.tolist() == ["XLE", "XLF", "XLK"]
    assert s.edge_index.shape == (2, 6)  # 3 pairs, both directions
    wide = prices.pivot(on="ticker", index="timestamp", values="close").sort("timestamp")
    rets = wide.select("XLE", "XLK").to_numpy()[-6:]
    rets = rets[1:] / rets[:-1] - 1
    expected = np.corrcoef(rets[:, 0], rets[:, 1])[0, 1]
    pair = (s.edge_index[0] == 0) & (s.edge_index[1] == 2)
    assert s.edge_attr[pair, 0] == pytest.approx(expected, abs=1e-6)
    assert s.edge_weight[pair] == pytest.approx(abs(expected), abs=1e-6)
    reverse = (s.edge_index[0] == 2) & (s.edge_index[1] == 0)
    assert s.edge_weight[reverse] == s.edge_weight[pair]


def test_x_is_the_feature_rows_at_t():
    prices = _prices()
    t = prices["timestamp"].sort()[-1]
    s = _builder(prices).build(t)
    feats = features.compute(prices, FEATURES_CFG).filter(pl.col("timestamp") == t).sort("ticker")
    assert s.feature_names.tolist() == ["momentum_5"]
    assert s.x[:, 0] == pytest.approx(feats["momentum_5"].to_numpy(), abs=1e-6)


def test_no_lookahead_truncating_the_future_changes_nothing():
    prices = _prices()
    t = prices["timestamp"].sort().unique()[12]
    full = _builder(prices).build(t)
    truncated = _builder(prices.filter(pl.col("timestamp") <= t)).build(t)
    for name in ("node_ids", "x", "edge_index", "edge_weight", "edge_attr"):
        assert np.array_equal(getattr(full, name), getattr(truncated, name))
