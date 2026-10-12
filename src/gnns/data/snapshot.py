"""GraphSnapshot: the one object graph builders emit. Contract: docs/CONTRACTS.md."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

TZ = "America/New_York"

# field -> (dtype kind, number of dimensions); "U" is a numpy unicode string array
_ARRAYS = {
    "node_ids": ("U", 1),
    "node_type": (np.int64, 1),
    "target_mask": (np.bool_, 1),
    "x": (np.float32, 2),
    "feature_names": ("U", 1),
    "feature_owner": (np.int64, 1),
    "edge_index": (np.int64, 2),
    "edge_weight": (np.float32, 1),
    "edge_type": (np.int64, 1),
    "edge_attr": (np.float32, 2),
    "edge_attr_names": ("U", 1),
}


@dataclass(frozen=True, eq=False)
class GraphSnapshot:
    timestamp: datetime
    node_ids: np.ndarray
    node_type: np.ndarray
    node_type_names: tuple[str, ...]
    target_mask: np.ndarray
    x: np.ndarray
    feature_names: np.ndarray
    feature_owner: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    edge_type: np.ndarray
    edge_type_names: tuple[str, ...]
    edge_attr: np.ndarray
    edge_attr_names: np.ndarray

    def __post_init__(self) -> None:
        _check_layout(self)
        _check_values(self)

    def save(self, path: Path) -> None:
        """Write to a .npz file; `load` restores an identical snapshot."""
        arrays = {name: getattr(self, name) for name in _ARRAYS}
        np.savez(
            path,
            **arrays,
            timestamp=np.array(self.timestamp.isoformat()),
            node_type_names=np.array(self.node_type_names, dtype=str),
            edge_type_names=np.array(self.edge_type_names, dtype=str),
        )

    @classmethod
    def load(cls, path: Path) -> "GraphSnapshot":
        with np.load(path, allow_pickle=False) as f:
            kwargs = {name: f[name] for name in _ARRAYS}
            ts = datetime.fromisoformat(str(f["timestamp"])).astimezone(ZoneInfo(TZ))
            return cls(
                timestamp=ts,
                node_type_names=tuple(f["node_type_names"].tolist()),
                edge_type_names=tuple(f["edge_type_names"].tolist()),
                **kwargs,
            )


def add_reverse_edges(
    edge_index: np.ndarray, edge_weight: np.ndarray, edge_type: np.ndarray, edge_attr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Append the reverse of every edge with the same weight, type and attributes.

    Builders pass each undirected edge once and get both directions back.
    """
    return (
        np.concatenate([edge_index, edge_index[::-1]], axis=1),
        np.concatenate([edge_weight, edge_weight]),
        np.concatenate([edge_type, edge_type]),
        np.concatenate([edge_attr, edge_attr]),
    )


def _check_layout(s: GraphSnapshot) -> None:
    """Types and shapes; raised before value checks, which assume them."""
    errors = []
    if s.timestamp.tzinfo is None or str(s.timestamp.tzinfo) != TZ:
        errors.append(f"timestamp must be tz-aware {TZ}, got {s.timestamp.tzinfo}")
    for name, (dtype, ndim) in _ARRAYS.items():
        a = getattr(s, name)
        if not isinstance(a, np.ndarray):
            errors.append(f"{name}: need a numpy array, got {type(a).__name__}")
            continue
        ok_dtype = a.dtype.kind == "U" if dtype == "U" else a.dtype == dtype
        if not ok_dtype or a.ndim != ndim:
            errors.append(f"{name}: need {ndim}-d {dtype}, got {a.ndim}-d {a.dtype}")
    if errors:
        raise ValueError("invalid GraphSnapshot:\n  " + "\n  ".join(errors))

    n, f = s.x.shape
    e, d = s.edge_attr.shape
    expected = {
        "node_ids": (n,),
        "node_type": (n,),
        "target_mask": (n,),
        "feature_names": (f,),
        "feature_owner": (f,),
        "edge_index": (2, e),
        "edge_weight": (e,),
        "edge_type": (e,),
        "edge_attr_names": (d,),
    }
    errors = [
        f"{name}: shape {getattr(s, name).shape}, expected {shape}"
        for name, shape in expected.items()
        if getattr(s, name).shape != shape
    ]
    if errors:
        raise ValueError("invalid GraphSnapshot:\n  " + "\n  ".join(errors))


def _check_values(s: GraphSnapshot) -> None:
    n_types, n_edge_types = len(s.node_type_names), len(s.edge_type_names)
    order = np.lexsort((s.node_ids, s.node_type))
    other_block = s.feature_owner[None, :] != s.node_type[:, None]
    checks = {
        "duplicate node_ids": len(set(s.node_ids)) != len(s.node_ids),
        "duplicate feature_names": len(set(s.feature_names)) != len(s.feature_names),
        "duplicate edge_attr_names": len(set(s.edge_attr_names)) != len(s.edge_attr_names),
        "node_type out of range": bool(((s.node_type < 0) | (s.node_type >= n_types)).any()),
        "feature_owner out of range": bool(
            ((s.feature_owner < 0) | (s.feature_owner >= n_types)).any()
        ),
        "edge_type out of range": bool(((s.edge_type < 0) | (s.edge_type >= n_edge_types)).any()),
        "nodes not sorted by (node_type, node_id)": not (order == np.arange(len(order))).all(),
        "x nonzero outside its node type's column block": bool(s.x[other_block].any()),
        "non-finite x": not np.isfinite(s.x).all(),
        "non-finite edge_weight": not np.isfinite(s.edge_weight).all(),
        "non-finite edge_attr": not np.isfinite(s.edge_attr).all(),
        "negative edge_weight": bool((s.edge_weight < 0).any()),
        "edge_index out of [0, N)": bool(
            ((s.edge_index < 0) | (s.edge_index >= len(s.node_ids))).any()
        ),
        "self-loops": bool((s.edge_index[0] == s.edge_index[1]).any()),
    }
    errors = [name for name, failed in checks.items() if failed]
    if errors:
        raise ValueError("invalid GraphSnapshot:\n  " + "\n  ".join(errors))
