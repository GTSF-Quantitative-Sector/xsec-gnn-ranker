"""Synthetic integration tests; no research dates or downloaded market inputs."""

import hashlib
import json
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
import torch
import yaml
from torch_geometric.loader import DataLoader

from gnns.data import features, store
from gnns.data.labels import forward_returns
from gnns.data.macro import FACTORS, UNITS, load_macro
from gnns.graphs.build import build_snapshots
from gnns.graphs.macro import MacroGraphBuilder
from gnns.train.dataset import build_dataset
from gnns.train.losses import listmle
from gnns.train.models import build_model


@pytest.fixture
def fixture(tmp_path):
    rng = np.random.default_rng(13)
    t0 = datetime(2018, 1, 1, 16, tzinfo=ZoneInfo("America/New_York"))
    ts = [t0 + timedelta(days=i) for i in range(110)]
    levels = np.array([3.0, 4.0, 20.0, 70.0, 120.0]) + np.cumsum(
        rng.normal(0, 0.1, (110, 5)), axis=0
    )
    rows = []
    for k, ticker in enumerate(["AAA", "BBB", "CCC"]):
        closes = 100 * np.cumprod(
            1 + rng.normal(0, 0.01, 110) + np.r_[0, np.diff(levels[:, k])] * 0.01
        )
        rows.extend(
            (t, ticker, float(c * 0.999), float(c * 1.01), float(c * 0.99), float(c), 100.0)
            for t, c in zip(ts, closes, strict=True)
        )
    prices = pl.DataFrame(rows, schema=store.PRICE_COLS, orient="row").sort("timestamp", "ticker")
    store.write(prices, tmp_path / "etfs" / "1d", {"universe": "etfs"})
    macro = pl.DataFrame(
        [
            (t.isoformat(), f, float(levels[i, k]))
            for i, t in enumerate(ts)
            for k, f in enumerate(FACTORS)
        ],
        schema=["timestamp", "factor", "value"],
        orient="row",
    )
    path = tmp_path / "macro.csv"
    macro.write_csv(path)
    manifest = dict(
        source="synthetic integration fixture",
        availability_method="synthetic bar-end release",
        known_at_verified=True,
        units=UNITS,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    path.with_suffix(".json").write_text(json.dumps(manifest))
    graph = yaml.safe_load(
        (Path(__file__).parents[2] / "config/graphs/macro_sensitivity.yaml").read_text()
    )
    graph.update(name="synthetic_macro", universe="etfs")
    graph["params"].update(window_bars=10, min_obs=8, ridge=1.0)
    graph["features"] = {"min_history_bars": 10, "list": [{"name": "momentum", "bars": 3}]}
    graph["sources"]["macro"]["path"] = "macro.csv"
    graph["model"].update(hidden=4, layers=1, heads=1, dropout=0.0)
    graph["baseline"].update(hidden=4, layers=1, dropout=0.0)
    cfg = dict(
        data_root=str(tmp_path),
        data={"bar": "1d"},
        graph=graph,
        normalize="none",
        horizon_bars=2,
        walk_forward=dict(
            window="rolling",
            train_bars=20,
            val_bars=8,
            test_bars=10,
            holdout_start=date(2018, 4, 1),
        ),
        train=dict(loss="listmle", lr=0.001, batch_size=8, max_epochs=2, patience=2, seed=7),
        eval=dict(quantiles=3, top_k=1, benchmark="SPY", rolling_dates=3),
        device="cpu",
    )
    return cfg, prices, ts, path


def builder(fixture, **params):
    cfg, prices, ts, _ = fixture
    return MacroGraphBuilder(
        {
            "prices": prices,
            "features": features.compute(prices, cfg["graph"]["features"]),
            "macro": load_macro(Path(cfg["data_root"]), cfg["graph"]["sources"]["macro"]),
        },
        pl.Series(ts),
        **(cfg["graph"]["params"] | params),
    )


def test_layout_mask_baseline_information_and_model_backward(fixture):
    cfg, prices, _, _ = fixture
    snapshots = build_snapshots(cfg)
    s = snapshots[0]
    assert s.node_ids.tolist() == [
        "AAA",
        "BBB",
        "CCC",
        "macro:credit",
        "macro:gold",
        "macro:oil",
        "macro:rates",
        "macro:vix",
    ]
    assert s.target_mask.tolist() == [True] * 3 + [False] * 5
    assert s.edge_index.shape == (2, 15)
    assert (s.edge_index[0] >= 3).all() and (s.edge_index[1] < 3).all()
    labels = forward_returns(prices, 2)
    data = build_dataset(snapshots[:3], labels, s.timestamp, snapshots[2].timestamp, cfg)
    assert data[0].label_mask.sum() == 3
    # Every context state is available on equity nodes to the ordinary MLP.
    for factor in FACTORS:
        k = s.feature_names.tolist().index(f"macro_change_{factor}")
        j = s.node_ids.tolist().index(f"macro:{factor}")
        context = s.feature_names.tolist().index("context_change")
        assert s.x[:3, k] == pytest.approx(np.repeat(s.x[j, context], 3))
    batch = next(iter(DataLoader(data[:2], batch_size=2)))
    for role in ("model", "baseline"):
        m = build_model(cfg["graph"][role], data[0])
        scores = m(batch)
        assert scores.shape == (16,)
        loss = listmle(scores, batch.y, batch.label_mask, batch.batch)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        m.eval()
        assert m(batch)[:8].detach().numpy() == pytest.approx(m(data[0]).detach().numpy(), abs=1e-6)


@pytest.mark.parametrize("mode", ["magnitude", "signed", "constant", "shuffled"])
def test_future_deletion_invariance(fixture, mode):
    b = builder(fixture, edge_mode=mode)
    t = fixture[2][40]
    full = b.build(t)
    trimmed = {name: df.filter(pl.col("timestamp") <= t) for name, df in b._inputs.items()}
    short = MacroGraphBuilder(
        trimmed,
        b._axis.filter(b._axis <= t),
        **(fixture[0]["graph"]["params"] | {"edge_mode": mode}),
    ).build(t)
    assert np.array_equal(full.x, short.x)
    assert np.array_equal(full.edge_attr, short.edge_attr)


def test_macro_not_seen_before_release_and_stale_rejected(fixture):
    b = builder(fixture, max_stale_bars=0)
    t = fixture[2][40]
    # A release after t cannot fill that day's missing factor.
    m = b._inputs["macro"]
    mask = (pl.col("timestamp") == t) & (pl.col("factor") == "credit")
    b._inputs["macro"] = m.with_columns(
        pl.when(mask)
        .then(pl.col("timestamp") + timedelta(hours=1))
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    ).sort("timestamp")
    assert not b.build(t).target_mask.any()


def test_constant_edges_remove_beta_from_both_channels_and_signed_preserves_sign(fixture):
    t = fixture[2][40]
    s = builder(fixture, edge_mode="constant").build(t)
    assert (s.edge_weight == 1).all() and (s.edge_attr == 1).all()
    s = builder(fixture, edge_mode="signed").build(t)
    idx = [s.feature_names.tolist().index(f"beta_{f}") for f in FACTORS]
    assert np.array_equal(s.edge_attr[:, 0], s.x[:3, idx].reshape(-1))
    assert np.array_equal(s.edge_weight, np.abs(s.edge_attr[:, 0]))


def test_missing_equity_price_never_forward_filled(fixture):
    b = builder(fixture, min_obs=10)
    t = fixture[2][40]
    b._inputs["prices"] = b._inputs["prices"].filter(
        ~((pl.col("ticker") == "BBB") & (pl.col("timestamp") == fixture[2][37]))
    )
    assert "BBB" not in b.build(t).node_ids


def test_roundtrip_and_unready_history(fixture, tmp_path):
    from gnns.data.snapshot import GraphSnapshot

    b = builder(fixture)
    assert not b.build(fixture[2][5]).target_mask.any()
    s = b.build(fixture[2][40])
    s.save(tmp_path / "snapshot.npz")
    loaded = GraphSnapshot.load(tmp_path / "snapshot.npz")
    assert np.array_equal(s.x, loaded.x)
    assert np.array_equal(s.edge_attr, loaded.edge_attr)


def test_additional_lag_preserves_beta_endpoint_but_keeps_current_equity_features(fixture):
    early = builder(fixture).build(fixture[2][38])
    lagged = builder(fixture, lag_bars=2).build(fixture[2][40])
    assert np.array_equal(early.edge_weight, lagged.edge_weight)
    assert np.array_equal(early.x[:3, 1:], lagged.x[:3, 1:])
    assert not np.array_equal(early.x[:3, 0], lagged.x[:3, 0])


def test_missing_offsets_and_duplicate_releases_rejected(fixture):
    cfg, _, _, path = fixture
    original = pl.read_csv(path)
    invalid = original.with_columns(pl.col("timestamp").str.slice(0, 19))
    invalid.write_csv(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="UTC offset"):
        load_macro(Path(cfg["data_root"]), {"path": path.name})
    pl.concat([original, original.head(1)]).write_csv(path)
    meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="duplicate"):
        load_macro(Path(cfg["data_root"]), {"path": path.name})


def test_requires_safe_normalization_and_verified_provenance(fixture):
    cfg, _, _, path = fixture
    cfg = deepcopy(cfg)
    cfg["normalize"] = "rank"
    with pytest.raises(ValueError, match="normalize: none"):
        build_snapshots(cfg)
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["known_at_verified"] = False
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="known-at"):
        load_macro(path.parent, {"path": path.name})
    meta["known_at_verified"] = True
    path.with_suffix(".json").write_text(json.dumps(meta))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_macro(path.parent, {"path": path.name})


def test_shared_run_logs_macro_provenance_predictions_and_reports(fixture, monkeypatch, tmp_path):
    import mlflow

    import gnns.config as config
    import gnns.graphs.build as build
    import gnns.train.run as runner
    from gnns.eval.report import report

    cfg, _, _, path = fixture
    for module in (config, build, runner):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "git_sha", lambda: "synthetic-fixture")
    run_id = runner.run(cfg)
    client = mlflow.MlflowClient()
    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert {
        "predictions_gatv2.parquet",
        "predictions_mlp.parquet",
        "source_manifest_macro.json",
        "config.yaml",
    } <= artifacts
    predictions = pl.read_parquet(
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="predictions_gatv2.parquet",
            dst_path=str(tmp_path / "readback"),
        )
    )
    assert predictions["timestamp"].max().date() < cfg["walk_forward"]["holdout_start"]
    assert not any(n.startswith("macro:") for n in predictions["ticker"])
    assert len(report(run_id, cfg["data_root"])) == 7


def test_local_config_resolves_construction_after_override(tmp_path, monkeypatch):
    import gnns.config as config

    (tmp_path / "graphs").mkdir()
    (tmp_path / "config.yaml").write_text("graph: original\nnormalize: rank\n")
    (tmp_path / "machine.yaml").write_text("graph: macro_sensitivity\nnormalize: none\n")
    (tmp_path / "graphs/macro_sensitivity.yaml").write_text("builder: macro_sensitivity\n")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    cfg = config.load_config()
    assert cfg["graph"]["builder"] == "macro_sensitivity"
    assert cfg["normalize"] == "none"
