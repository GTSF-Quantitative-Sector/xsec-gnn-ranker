from dataclasses import replace
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from gnns.train.dataset import build_dataset, normalize, to_data
from tests.data.test_snapshot import _snapshot  # AAPL, XOM (equity) + crude (commodity)

COLS = ["mom_20", "vol_20", "px_level"]


def _labels(s, rows: dict[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"timestamp": [s.timestamp] * len(rows), "ticker": list(rows), "fwd_return": rows.values()}
    )


def test_rank_scales_ties_and_lone_nodes():
    v = np.array([[3.0], [1.0], [2.0], [2.0]], dtype=np.float32)
    out = normalize(v, np.zeros(4, dtype=np.int64), np.zeros(1, dtype=np.int64), "rank")
    assert out[:, 0].tolist() == pytest.approx([1, -1, 0, 0])
    lone = normalize(v[:1], np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), "rank")
    assert lone.tolist() == [[0.0]]


def test_zscore_clips_and_handles_no_spread():
    v = np.array([[0.0]] * 10 + [[100.0]], dtype=np.float32)
    out = normalize(v, np.zeros(11, dtype=np.int64), np.zeros(1, dtype=np.int64), "zscore")
    assert out.max() == pytest.approx(3.0)
    flat = normalize(
        np.ones((3, 1), dtype=np.float32),
        np.zeros(3, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        "zscore",
    )
    assert flat.tolist() == [[0.0]] * 3


def test_normalizes_within_each_node_type_and_keeps_column_blocks():
    s = _snapshot()
    d = to_data(s, None, COLS, [], "rank")
    # Equities ranked among themselves; the lone commodity gets 0; padding stays 0.
    assert d["x"].tolist() == [[-1, -1, 0], [1, 1, 0], [0, 0, 0]]


def test_labels_and_mask_keep_every_node():
    s = _snapshot()
    d = to_data(s, _labels(s, {"AAPL": 0.02}), COLS, [], "none")
    assert d.node_ids == ["AAPL", "XOM", "crude"]  # XOM has no label but stays in the graph
    assert d["y"].tolist() == pytest.approx([0.02, 0.0, 0.0])
    assert d["label_mask"].tolist() == [True, False, False]  # crude is never ranked
    assert d["edge_index"].shape == (2, 2)


def test_selects_only_listed_columns():
    s = _snapshot()
    d = to_data(s, None, ["vol_20"], ["sign"], "none")
    assert d["x"][:, 0].tolist() == pytest.approx([0.2, 0.4, 0.0])
    assert d["edge_attr"].tolist() == [[-1.0], [1.0]]
    assert to_data(s, None, ["vol_20"], [], "none")["edge_attr"].shape == (2, 0)
    with pytest.raises(ValueError, match="not in snapshot"):
        to_data(s, None, ["momentun_20"], [], "none")


def test_build_dataset_keeps_snapshots_in_the_period():
    a = _snapshot()
    b = replace(a, timestamp=a.timestamp + timedelta(days=7))
    cfg = {
        "graph": {
            "features": {"list": [{"name": "mom", "bars": 20}]},
            "extra_model_features": ["vol_20"],
        },
        "normalize": "rank",
    }
    data = build_dataset([a, b], _labels(b, {"XOM": 0.01}), b.timestamp, b.timestamp, cfg)
    assert len(data) == 1 and data[0].timestamp == b.timestamp
    assert data[0]["x"].shape == (3, 2)
    assert data[0]["label_mask"].tolist() == [False, True, False]


def test_bad_normalize_raises():
    with pytest.raises(ValueError, match="normalize must be"):
        to_data(_snapshot(), None, COLS, [], "minmax")
