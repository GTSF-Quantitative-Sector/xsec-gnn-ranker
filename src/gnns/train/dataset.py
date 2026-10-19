"""GraphSnapshotDataset: snapshots + labels for one period -> PyG `Data` samples.

One sample is one snapshot (the whole cross-section at t). Which columns reach the model:
docs/CONTRACTS.md, "What reaches the model". Decisions: src/gnns/train/CLAUDE.md.
"""

from datetime import datetime

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

from gnns.data.features import column_names
from gnns.data.snapshot import GraphSnapshot


def model_columns(graph: dict) -> tuple[list[str], list[str]]:
    """(node columns, edge columns) the model receives, from a construction's graph config."""
    nodes = column_names(graph["features"]["list"]) + graph.get("extra_model_features", [])
    return nodes, graph.get("model_edge_attrs", [])


def build_dataset(
    snapshots: list[GraphSnapshot],
    labels: pl.DataFrame,
    start: datetime,
    end: datetime,
    cfg: dict,
) -> list[Data]:
    """One `Data` per snapshot with start <= timestamp <= end (inclusive, e.g. a fold period)."""
    node_cols, edge_cols = model_columns(cfg["graph"])
    by_t = labels.partition_by("timestamp", as_dict=True)
    return [
        to_data(s, by_t.get((s.timestamp,)), node_cols, edge_cols, cfg["normalize"])
        for s in snapshots
        if start <= s.timestamp <= end
    ]


def to_data(
    s: GraphSnapshot,
    labels_t: pl.DataFrame | None,
    node_cols: list[str],
    edge_cols: list[str],
    how: str,
) -> Data:
    """Select model columns, normalize them, and attach labels; nodes are never dropped."""
    cols = _indices(s.feature_names, node_cols, "node")
    x = normalize(s.x[:, cols], s.node_type, s.feature_owner[cols], how)
    edge_attr = s.edge_attr[:, _indices(s.edge_attr_names, edge_cols, "edge")]

    fwd = (
        {}
        if labels_t is None
        else dict(zip(labels_t["ticker"], labels_t["fwd_return"], strict=True))
    )
    y = np.array([fwd.get(n, np.nan) for n in s.node_ids], dtype=np.float32)
    # Ranked nodes with a label count in the loss and IC; the rest still pass messages.
    label_mask = s.target_mask & np.isfinite(y)
    return Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(s.edge_index),
        edge_weight=torch.from_numpy(s.edge_weight),
        edge_attr=torch.from_numpy(np.ascontiguousarray(edge_attr)),
        y=torch.from_numpy(np.nan_to_num(y)),
        label_mask=torch.from_numpy(label_mask),
        node_ids=s.node_ids.tolist(),
        timestamp=s.timestamp,
    )


def normalize(x: np.ndarray, node_type: np.ndarray, owner: np.ndarray, how: str) -> np.ndarray:
    """Scale each column across the nodes of the type that owns it, at one t."""
    if how not in _SCALERS:
        raise ValueError(f"normalize must be one of {sorted(_SCALERS)}, got {how!r}")
    out = x.copy()
    for t in np.unique(owner):
        block = np.ix_(node_type == t, owner == t)
        if out[block].size:
            out[block] = np.apply_along_axis(_SCALERS[how], 0, x[block])
    return out


def _rank(v: np.ndarray) -> np.ndarray:
    """Percentile rank scaled to [-1, 1]; ties share their average rank; a lone node gets 0."""
    if len(v) < 2:
        return np.zeros_like(v)
    r = np.empty(len(v))
    r[np.argsort(v, kind="stable")] = np.arange(1, len(v) + 1)
    _, group = np.unique(v, return_inverse=True)
    r = (np.bincount(group, r) / np.bincount(group))[group]
    return 2 * (r - 1) / (len(v) - 1) - 1


def _zscore(v: np.ndarray) -> np.ndarray:
    """(v - mean) / std clipped to ±3; 0 when there is no spread."""
    sd = v.std()
    return np.zeros_like(v) if sd == 0 else np.clip((v - v.mean()) / sd, -3, 3)


_SCALERS = {"rank": _rank, "zscore": _zscore, "none": lambda v: v}


def _indices(names: np.ndarray, wanted: list[str], kind: str) -> np.ndarray:
    pos = {n: i for i, n in enumerate(names)}
    if missing := [w for w in wanted if w not in pos]:
        raise ValueError(f"{kind} columns {missing} not in snapshot; available: {list(names)}")
    return np.array([pos[w] for w in wanted], dtype=np.int64)
