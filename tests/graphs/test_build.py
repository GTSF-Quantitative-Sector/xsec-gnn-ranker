import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from gnns.data import store
from gnns.data.snapshot import GraphSnapshot
from gnns.graphs.base import GraphBuilder
from gnns.graphs.build import BUILDERS, build_snapshots
from tests.data.test_store import MANIFEST

T0 = datetime(2024, 1, 1, 16, tzinfo=ZoneInfo("America/New_York"))


def _cfg(tmp_path, window_bars: int = 5, min_history_bars: int = 5) -> dict:
    rng = np.random.default_rng(2)
    rows = [
        (T0 + timedelta(days=i), k, float(c), float(c), float(c), float(c), 100.0)
        for k in ("XLE", "XLF", "XLK")
        for i, c in enumerate(100 * np.cumprod(1 + rng.normal(0, 0.02, 30)))
    ]
    prices = pl.DataFrame(rows, schema=store.PRICE_COLS, orient="row")
    dest = tmp_path / "etfs" / "1d"
    store.write(prices.sort("timestamp", "ticker"), dest, MANIFEST | {"universe": "etfs"})
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
        "horizon_bars": 5,
    }


def test_snapshots_every_k_bars_once_nodes_are_eligible(tmp_path):
    snaps, _ = build_snapshots(_cfg(tmp_path))
    # Bars 0..29; positions 0, 5, ..., 25; eligibility starts at position 5.
    assert [s.timestamp.day for s in snaps] == [6, 11, 16, 21, 26]
    assert all(s.edge_index.shape == (2, 6) for s in snaps)


def test_history_shorter_than_price_window_raises(tmp_path):
    with pytest.raises(ValueError, match="partial windows"):
        build_snapshots(_cfg(tmp_path, window_bars=10, min_history_bars=5))


def test_sector_etf_build_returns_the_price_manifest_and_drops_nothing(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="gnns.graphs.build")
    _, manifests = build_snapshots(_cfg(tmp_path))
    on_disk = json.loads((tmp_path / "etfs" / "1d" / "manifest.json").read_text())
    assert manifests == {"prices": on_disk}
    assert "dropped" not in caplog.text


# Rows known at Jan 3, Jan 13 (twice) and after the price range.
EVENTS = pl.DataFrame(
    {
        "timestamp": [
            T0 + timedelta(days=2),
            T0 + timedelta(days=12),
            T0 + timedelta(days=12, hours=3),
            T0 + timedelta(days=60),
        ]
    }
)


class Toy(GraphBuilder):
    """One target node per eligible ticker (none on `empty_days`); its one column is the
    number of `events` rows known at t, which it marks prescaled."""

    def __init__(self, inputs, axis, empty_days=()):
        super().__init__(inputs, axis)
        self.empty_days = set(empty_days)

    @classmethod
    def check_config(cls, cfg):
        if cfg["normalize"] != "rank":
            raise ValueError("toy needs normalize: rank")

    @classmethod
    def load_sources(cls, root, sources):
        lf, manifest = store.open(root / sources["events"]["path"])
        return {"events": (lf.collect(), manifest)}

    def lookbacks(self):
        return {"features": 1, "events": None}

    def build_at(self, t, inputs):
        nodes = inputs["features"]["ticker"].sort().to_list()
        n = len(nodes)
        return GraphSnapshot(
            timestamp=t,
            node_ids=np.array(nodes, dtype=str),
            node_type=np.zeros(n, dtype=np.int64),
            node_type_names=("equity",),
            target_mask=np.full(n, t.day not in self.empty_days),
            x=np.full((n, 1), inputs["events"].height, dtype=np.float32),
            feature_names=np.array(["n_events"]),
            feature_owner=np.zeros(1, dtype=np.int64),
            feature_prescaled=np.ones(1, dtype=np.bool_),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_weight=np.zeros(0, dtype=np.float32),
            edge_type=np.zeros(0, dtype=np.int64),
            edge_type_names=("none",),
            edge_attr=np.zeros((0, 0), dtype=np.float32),
            edge_attr_names=np.array([], dtype=str),
        )


def _toy_cfg(tmp_path, monkeypatch, **params) -> dict:
    monkeypatch.setitem(BUILDERS, "toy", Toy)
    cfg = _cfg(tmp_path)
    store.write(EVENTS, tmp_path / "events", MANIFEST | {"kind": "toy"}, file="events.parquet")
    sources = {"events": {"path": "events"}}
    cfg["graph"] |= {"builder": "toy", "params": params, "sources": sources}
    return cfg | {"normalize": "rank"}


def test_builder_sources_reach_the_builder_and_their_manifests_are_returned(tmp_path, monkeypatch):
    snaps, manifests = build_snapshots(_toy_cfg(tmp_path, monkeypatch))
    # Point-in-time: at each t the builder saw only the events known by t.
    assert [s.timestamp.day for s in snaps] == [6, 11, 16, 21, 26]
    assert [int(s.x[0, 0]) for s in snaps] == [1, 1, 3, 3, 3]
    assert all(s.feature_prescaled.tolist() == [True] for s in snaps)
    assert manifests.keys() == {"prices", "events"}
    on_disk = json.loads((tmp_path / "events" / "manifest.json").read_text())
    assert manifests["events"] == on_disk


def test_check_config_runs_before_any_data_is_loaded(tmp_path, monkeypatch):
    cfg = _toy_cfg(tmp_path, monkeypatch) | {"normalize": "zscore"}
    cfg["data_root"] = str(tmp_path / "nothing_built_here")
    with pytest.raises(ValueError, match="toy needs normalize: rank"):
        build_snapshots(cfg)


def test_reserved_source_names_raise(tmp_path, monkeypatch):
    cfg = _toy_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(Toy, "load_sources", classmethod(lambda cls, root, s: {"prices": None}))
    with pytest.raises(ValueError, match=r"source names \['prices'\] are reserved"):
        build_snapshots(cfg)


def test_dates_without_a_target_node_are_dropped_and_logged(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="gnns.graphs.build")
    snaps, _ = build_snapshots(_toy_cfg(tmp_path, monkeypatch, empty_days=[11, 21]))
    assert [s.timestamp.day for s in snaps] == [6, 16, 26]
    assert (
        "toy: dropped 2 of 5 dates with no target node "
        "(2024-01-11 16:00:00-05:00 .. 2024-01-21 16:00:00-05:00)"
    ) in caplog.text


def test_no_target_node_on_any_date_raises(tmp_path, monkeypatch):
    cfg = _toy_cfg(tmp_path, monkeypatch, empty_days=[6, 11, 16, 21, 26])
    with pytest.raises(ValueError, match="toy: no snapshot has a target node"):
        build_snapshots(cfg)
