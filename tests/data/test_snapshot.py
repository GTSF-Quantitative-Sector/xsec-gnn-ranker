import re
from dataclasses import replace
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from gnns.data.snapshot import GraphSnapshot, add_reverse_edges


def _snapshot(**overrides: Any) -> GraphSnapshot:
    """Two equities with 2 features, one commodity with 1, in column blocks."""
    kwargs: dict[str, Any] = dict(
        timestamp=datetime(2024, 1, 2, 16, tzinfo=ZoneInfo("America/New_York")),
        node_ids=np.array(["AAPL", "XOM", "crude"]),
        node_type=np.array([0, 0, 1]),
        node_type_names=("equity", "commodity"),
        target_mask=np.array([True, True, False]),
        x=np.array([[0.1, 0.2, 0], [0.3, 0.4, 0], [0, 0, 78.2]], dtype=np.float32),
        feature_names=np.array(["mom_20", "vol_20", "px_level"]),
        feature_owner=np.array([0, 0, 1]),
        edge_index=np.array([[2, 0], [1, 1]]),
        edge_weight=np.array([0.5, 1.0], dtype=np.float32),
        edge_type=np.array([1, 0]),
        edge_type_names=("correlated", "exposes"),
        edge_attr=np.array([[-1.0], [1.0]], dtype=np.float32),
        edge_attr_names=np.array(["sign"]),
    )
    return GraphSnapshot(**(kwargs | overrides))


def test_valid_snapshot_constructs():
    assert _snapshot().x.shape == (3, 3)


def test_save_then_load_is_identical(tmp_path):
    s = _snapshot()
    s.save(tmp_path / "s.npz")
    loaded = GraphSnapshot.load(tmp_path / "s.npz")
    assert loaded.timestamp == s.timestamp
    assert str(loaded.timestamp.tzinfo) == "America/New_York"
    assert loaded.node_type_names == s.node_type_names
    assert loaded.edge_type_names == s.edge_type_names
    for name in ("node_ids", "x", "edge_index", "edge_attr", "edge_attr_names", "target_mask"):
        a, b = getattr(loaded, name), getattr(s, name)
        assert a.dtype == b.dtype and np.array_equal(a, b)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"x": np.zeros((3, 3))}, "x: need 2-d"),  # float64
        ({"node_ids": np.array(["AAPL", "XOM"])}, "node_ids: shape"),
        ({"timestamp": datetime(2024, 1, 2, 16)}, "tz-aware"),
        ({"node_ids": np.array(["XOM", "AAPL", "crude"])}, "not sorted"),
        ({"node_ids": np.array(["AAPL", "AAPL", "crude"])}, "duplicate node_ids"),
        ({"node_type": np.array([0, 0, 2])}, "node_type out of range"),
        ({"x": np.ones((3, 3), dtype=np.float32)}, "column block"),
        ({"x": np.array([[np.nan, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float32)}, "non-finite x"),
        ({"edge_weight": np.array([-0.5, 1.0], dtype=np.float32)}, "negative edge_weight"),
        ({"edge_index": np.array([[2, 0], [1, 3]])}, "out of [0, N)"),
        ({"edge_index": np.array([[2, 1], [1, 1]])}, "self-loops"),
    ],
)
def test_invalid_snapshot_raises(overrides, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        _snapshot(**overrides)


def test_feature_prescaled_defaults_to_none_prescaled():
    p = _snapshot().feature_prescaled
    assert p.dtype == np.bool_ and p.tolist() == [False, False, False]


def test_feature_prescaled_round_trips(tmp_path):
    s = _snapshot(feature_prescaled=np.array([False, True, False]))
    s.save(tmp_path / "s.npz")
    assert GraphSnapshot.load(tmp_path / "s.npz").feature_prescaled.tolist() == [False, True, False]


def test_file_saved_before_feature_prescaled_loads_as_none_prescaled(tmp_path):
    _snapshot().save(tmp_path / "new.npz")
    with np.load(tmp_path / "new.npz") as f:
        old = {k: f[k] for k in f.files if k != "feature_prescaled"}
    np.savez(tmp_path / "old.npz", **old)
    loaded = GraphSnapshot.load(tmp_path / "old.npz")
    assert loaded.feature_prescaled.tolist() == [False, False, False]


def test_feature_prescaled_needs_one_flag_per_column():
    with pytest.raises(ValueError, match=re.escape("feature_prescaled: shape (2,)")):
        _snapshot(feature_prescaled=np.array([True, False]))


def test_replace_revalidates():
    with pytest.raises(ValueError, match="self-loops"):
        replace(_snapshot(), edge_index=np.array([[0, 0], [0, 1]]))


def test_add_reverse_edges_mirrors_everything():
    ei, w, t, a = add_reverse_edges(
        np.array([[0], [1]]),
        np.array([0.5], dtype=np.float32),
        np.array([0]),
        np.array([[-1.0]], dtype=np.float32),
    )
    assert ei.tolist() == [[0, 1], [1, 0]]
    assert w.tolist() == [0.5, 0.5] and t.tolist() == [0, 0] and a.tolist() == [[-1.0], [-1.0]]
