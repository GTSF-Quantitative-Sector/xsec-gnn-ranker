from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
import yaml

from gnns.config import CONFIG_DIR
from gnns.data import features, store
from gnns.graphs.oil_supply_chain import OilSupplyChainBuilder
from gnns.train import dataset, run
from gnns.train.models import build_model
from tests.data.test_store import MANIFEST


@pytest.fixture
def oil_panel():
    graph = yaml.safe_load((CONFIG_DIR / "graphs" / "oil_supply_chain.yaml").read_text())
    tickers = (CONFIG_DIR / "universes" / "oil_supply_chain.txt").read_text().split()
    start = datetime(2023, 1, 1, 16, tzinfo=ZoneInfo("America/New_York"))
    rng = np.random.default_rng(17)
    rows = [
        (start + timedelta(days=i), ticker, close, close, close, close, 100.0)
        for ticker in tickers
        for i, close in enumerate(100 * np.cumprod(1 + rng.normal(0, 0.01, 150)))
    ]
    prices = pl.DataFrame(rows, schema=store.PRICE_COLS, orient="row").sort("timestamp", "ticker")
    return graph, prices


def _snapshot(graph, prices, t):
    axis = prices["timestamp"].unique().sort()
    inputs = {"prices": prices, "features": features.compute(prices, graph["features"])}
    return OilSupplyChainBuilder(inputs, axis, **graph["params"]).build(t)


def test_oil_paths_masks_and_model_inputs(oil_panel):
    graph, prices = oil_panel
    axis = prices["timestamp"].unique().sort()
    snap = _snapshot(graph, prices, axis[70])
    edges = {(snap.node_ids[s], snap.node_ids[d]) for s, d in snap.edge_index.T}
    assert len(edges) == 13
    assert {("BNO", "XOM"), ("BNO", "MPC"), ("MPC", "DAL"), ("VLO", "ODFL")} <= edges
    assert all("NEE" not in pair for pair in edges)
    assert snap.node_ids[~snap.target_mask].tolist() == ["BNO"]
    oil = prices.filter(pl.col("ticker") == "BNO")["close"]
    expected = oil[70] / oil[65] - 1
    assert snap.x[0, list(snap.feature_names).index("oil_return_5")] == pytest.approx(expected)
    assert snap.x[-1, list(snap.feature_names).index("oil_context_return_5")] == pytest.approx(
        expected
    )
    nodes, attrs = dataset.model_columns(graph)
    sample = dataset.to_data(snap, None, nodes, attrs, "rank")
    assert sample.x[0, nodes.index("oil_return_5")].item() == pytest.approx(expected)
    np.testing.assert_array_equal(sample.edge_attr.numpy().sum(axis=1), 1)
    for role in ("model", "baseline"):
        scores = build_model(graph[role], sample)(sample)
        assert scores.shape == (11,)
        scores.sum().backward()


def test_missing_bar_does_not_depend_on_future_presence(oil_panel):
    graph, prices = oil_panel
    t = prices["timestamp"].unique().sort()[70]
    prices = prices.filter(~((pl.col("timestamp") == t) & (pl.col("ticker") == "DAL")))
    full = _snapshot(graph, prices, t)
    truncated = _snapshot(graph, prices.filter(pl.col("timestamp") <= t), t)
    assert "DAL" not in full.node_ids
    np.testing.assert_array_equal(full.node_ids, truncated.node_ids)
    np.testing.assert_array_equal(full.edge_index, truncated.edge_index)
    np.testing.assert_array_equal(full.x, truncated.x)
    no_bno_at_t = prices.filter(~((pl.col("timestamp") == t) & (pl.col("ticker") == "BNO")))
    assert not _snapshot(graph, no_bno_at_t, t).target_mask.any()


def test_oil_pipeline_trains_both_models_with_search_only_labels(oil_panel, tmp_path):
    graph, prices = oil_panel
    axis = prices["timestamp"].unique().sort()
    holdout = axis[125].date()
    store.write(prices, tmp_path / graph["universe"] / "1d", MANIFEST)
    cfg = {
        "data_root": str(tmp_path),
        "data": {"bar": "1d"},
        "graph": graph,
        "horizon_bars": 2,
        "normalize": "rank",
        "walk_forward": {
            "window": "rolling",
            "train_bars": 24,
            "val_bars": 8,
            "test_bars": 12,
            "holdout_start": holdout,
        },
        "train": {
            "seed": 0,
            "lr": 0.01,
            "batch_size": 8,
            "loss": "listmle",
            "max_epochs": 2,
            "patience": 2,
        },
    }
    built = run.build(cfg)
    assert built.folds
    assert max(d.timestamp for fold in built.data for d in fold["test"]) == axis[120]
    for role in ("model", "baseline"):
        result = run.train(built, role, cfg, "cpu")
        assert result.predictions.height
        assert "BNO" not in result.predictions["ticker"]
        assert np.isfinite(result.predictions["score"].to_numpy()).all()
