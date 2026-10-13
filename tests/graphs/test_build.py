from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from gnns.data import store
from gnns.graphs.build import build_snapshots


def _cfg(tmp_path, window_bars: int = 5, min_history_bars: int = 5) -> dict:
    rng = np.random.default_rng(2)
    t0 = datetime(2024, 1, 1, 16, tzinfo=ZoneInfo("America/New_York"))
    rows = [
        (t0 + timedelta(days=i), k, float(c), float(c), float(c), float(c), 100.0)
        for k in ("XLE", "XLF", "XLK")
        for i, c in enumerate(100 * np.cumprod(1 + rng.normal(0, 0.02, 30)))
    ]
    prices = pl.DataFrame(rows, schema=store.PRICE_COLS, orient="row")
    store.write(prices.sort("timestamp", "ticker"), tmp_path / "etfs" / "1d", {"universe": "etfs"})
    return {
        "data_root": str(tmp_path),
        "data": {"bar": "1d"},
        "graph": {
            "name": "test",
            "builder": "sector_etf_correlation",
            "universe": "etfs",
            "params": {"window_bars": window_bars},
            "features": {
                "min_history_bars": min_history_bars,
                "list": [{"name": "momentum", "bars": 5}],
            },
        },
        "snapshot_every_bars": 5,
    }


def test_snapshots_every_k_bars_once_nodes_are_eligible(tmp_path):
    snaps = build_snapshots(_cfg(tmp_path))
    # Bars 0..29; positions 0, 5, ..., 25; eligibility starts at position 5.
    assert [s.timestamp.day for s in snaps] == [6, 11, 16, 21, 26]
    assert all(s.edge_index.shape == (2, 6) for s in snaps)


def test_history_shorter_than_price_window_raises(tmp_path):
    with pytest.raises(ValueError, match="partial windows"):
        build_snapshots(_cfg(tmp_path, window_bars=10, min_history_bars=5))
