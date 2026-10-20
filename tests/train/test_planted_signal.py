"""Planted-signal tests: synthetic data with a known answer, run through the real Dataset,
models, ListMLE and Trainer. If these fail, the training code is broken; if they pass, a
zero IC on real data means the inputs carry no learnable signal.

Each snapshot has `N` nodes with `F` features x ~ N(0, 1). Labels are built from x with a
chosen amount of noise, so the best achievable per-date rank IC is known in advance:
for y = x0 + sigma * e, Pearson rho = 1 / sqrt(1 + sigma^2) and Spearman
rho_s = (6 / pi) * asin(rho / 2).
"""

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
import torch

from gnns.data.snapshot import GraphSnapshot, add_reverse_edges
from gnns.eval.importance import permutation_importance
from gnns.eval.metrics import daily_ic, ic_summary
from gnns.train.dataset import build_dataset
from gnns.train.losses import listmle, no_info_loss
from gnns.train.models import build_model
from gnns.train.trainer import predict, train_fold

N, F = 10, 3
T0 = datetime(2010, 1, 1, 16, tzinfo=ZoneInfo("America/New_York"))
TRAIN = {"lr": 0.01, "batch_size": 16, "max_epochs": 60, "patience": 60, "seed": 0}


def expected_spearman(sigma: float) -> float:
    """Population rank IC between x0 and y = x0 + sigma * e, both Gaussian."""
    return 6 / math.pi * math.asin(1 / math.sqrt(1 + sigma**2) / 2)


def _edges(rng: np.random.Generator, dense: bool) -> np.ndarray:
    """Fully connected (like the sector-ETF graph), or sparse: each node links to 2 others."""
    if dense:
        pairs = {(i, j) for i in range(N) for j in range(i + 1, N)}
    else:
        pairs = set()
        for i in range(N):
            for j in rng.choice([k for k in range(N) if k != i], size=2, replace=False):
                pairs.add((min(i, int(j)), max(i, int(j))))
    return np.array(sorted(pairs), dtype=np.int64).T


def make_data(n_dates: int, sigma: float, signal: str, dense: bool, seed: int):
    """Synthetic snapshots + labels through the real Dataset code.

    signal: "own" -> y_i = x0_i + noise; "neighbors" -> y_i = mean of x0 over i's
    neighbors + noise (only a graph model can see it); "none" -> y_i = noise.
    """
    rng = np.random.default_rng(seed)
    ids = np.array([f"N{i:02d}" for i in range(N)])
    snaps, rows = [], []
    for d in range(n_dates):
        t = T0 + timedelta(days=7 * d)
        x = rng.normal(size=(N, F)).astype(np.float32)
        ei, w, et, ea = add_reverse_edges(
            ei := _edges(rng, dense),
            np.ones(ei.shape[1], dtype=np.float32),
            np.zeros(ei.shape[1], dtype=np.int64),
            np.zeros((ei.shape[1], 0), dtype=np.float32),
        )
        if signal == "own":
            clean = x[:, 0]
        elif signal == "neighbors":  # mean of x0 over in-neighbors (messages flow src -> dst)
            clean = np.array([x[ei[0][ei[1] == i], 0].mean() for i in range(N)])
        else:
            clean = np.zeros(N)
        y = clean + sigma * rng.normal(size=N)
        snaps.append(
            GraphSnapshot(
                timestamp=t,
                node_ids=ids,
                node_type=np.zeros(N, dtype=np.int64),
                node_type_names=("equity",),
                target_mask=np.ones(N, dtype=bool),
                x=x,
                feature_names=np.array([f"f_{k}" for k in range(F)]),
                feature_owner=np.zeros(F, dtype=np.int64),
                edge_index=ei,
                edge_weight=w,
                edge_type=et,
                edge_type_names=("link",),
                edge_attr=ea,
                edge_attr_names=np.array([], dtype=str),
            )
        )
        rows += [(t, n, float(v)) for n, v in zip(ids, y, strict=True)]
    labels = pl.DataFrame(rows, schema=["timestamp", "ticker", "fwd_return"], orient="row")
    cfg = {
        "graph": {"features": {"list": [{"name": "f", "k": k} for k in range(F)]}},
        "normalize": "rank",
    }
    return build_dataset(snaps, labels, snaps[0].timestamp, snaps[-1].timestamp, cfg)


def train_model(model_name: str, data, n_train: int, n_val: int, train_cfg: dict = TRAIN):
    """Train on the first n_train samples, early-stop on the next n_val."""
    torch.manual_seed(0)
    train, val = data[:n_train], data[n_train : n_train + n_val]
    spec = {"name": model_name, "hidden": 32, "layers": 2}
    return train_fold(build_model(spec, train[0]), train, val, train_cfg, listmle, "cpu")


def fit(model_name: str, data, n_train: int, n_val: int, train_cfg: dict = TRAIN) -> dict:
    """Train (as above), then score the samples after the validation ones."""
    model, hist = train_model(model_name, data, n_train, n_val, train_cfg)
    train, test = data[:n_train], data[n_train + n_val :]
    return {
        "loss": min(e["train_loss"] for e in hist),
        "no_info": no_info_loss("listmle", train),
        "test": ic_summary(daily_ic(predict(model, test, "cpu"))),
    }


# 240 weekly dates ≈ one walk-forward fold: 150 train, 30 validation, 60 test.


@pytest.mark.parametrize("model", ["mlp", "gatv2"])
def test_strong_signal_is_learned(model):
    """sigma = 1: best possible rank IC 0.69. On a fully connected graph, like sector ETFs."""
    r = fit(model, make_data(240, 1.0, "own", dense=True, seed=0), 150, 30)
    assert r["loss"] < r["no_info"] - 0.1  # clearly below guessing
    assert r["test"]["mean_ic"] > 0.5


def test_moderate_signal_is_learned_at_one_fold_of_data():
    """sigma = 3: best possible rank IC 0.30, learnable from one fold's worth of dates."""
    r = fit("mlp", make_data(240, 3.0, "own", dense=False, seed=0), 150, 30)
    assert r["test"]["mean_ic"] > 0.15 and r["test"]["ic_t"] > 3


def test_weak_signal_needs_many_dates():
    """sigma = 10: best possible rank IC 0.095, about realistic for equities. Recovered from
    ~900 training dates, though the training loss barely moves below guessing."""
    r = fit("mlp", make_data(1500, 10.0, "own", dense=False, seed=0), 900, 150)
    assert r["test"]["mean_ic"] > 0.05 and r["test"]["ic_t"] > 3
    assert r["loss"] > r["no_info"] - 0.01  # learned, yet invisible in the loss curve


def test_neighbor_signal_needs_the_graph():
    """y_i = mean of x0 over i's neighbors: GATv2 sees it; the MLP only gets a small
    cross-sectional side effect (each node's x0 lifts its neighbors' labels)."""
    data = make_data(240, 0.3, "neighbors", dense=False, seed=0)
    gat, mlp = fit("gatv2", data, 150, 30), fit("mlp", data, 150, 30)
    assert gat["test"]["mean_ic"] > 0.5
    assert gat["test"]["mean_ic"] > mlp["test"]["mean_ic"] + 0.3  # detectable graph lift


@pytest.mark.parametrize("model", ["mlp", "gatv2"])
def test_no_signal_is_not_invented(model):
    """Pure-noise labels: loss stays at guessing and test IC stays near zero (no leakage)."""
    r = fit(model, make_data(240, 1.0, "none", dense=False, seed=0), 150, 30)
    assert r["loss"] > r["no_info"] - 0.06
    assert abs(r["test"]["mean_ic"]) < 0.2 and abs(r["test"]["ic_t"]) < 3


COLUMNS = [f"f_{k}" for k in range(F)]


def test_permutation_importance_finds_the_planted_feature():
    """y = x0 + noise: scrambling f_0 destroys the IC; f_1, f_2 and edges barely matter."""
    data = make_data(240, 1.0, "own", dense=False, seed=0)
    model, _ = train_model("mlp", data, 150, 30)
    imp = permutation_importance(model, data[150:], COLUMNS, repeats=3, device="cpu")
    assert imp["f_0"] > 0.4
    assert abs(imp["f_1"]) < 0.1 and abs(imp["f_2"]) < 0.1
    assert imp["edges"] == 0  # the MLP never reads edges


def test_permutation_importance_finds_the_planted_graph_signal():
    """y = mean of neighbors' x0: rewiring edges destroys GATv2's IC."""
    data = make_data(240, 0.3, "neighbors", dense=False, seed=0)
    model, _ = train_model("gatv2", data, 150, 30)
    imp = permutation_importance(model, data[150:], COLUMNS, repeats=3, device="cpu")
    assert imp["edges"] > 0.4
    assert imp["f_0"] > 0.4  # it is the neighbors' f_0 that matters
    assert abs(imp["f_1"]) < 0.1 and abs(imp["f_2"]) < 0.1
