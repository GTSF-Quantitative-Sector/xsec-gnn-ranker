"""Synthetic integration tests; no research dates or downloaded market inputs."""

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
from gnns.graphs.macro import PRESCALED, MacroGraphBuilder
from gnns.train.dataset import build_dataset, model_columns
from gnns.train.losses import listmle
from gnns.train.models import build_model
from tests.data.test_store import MANIFEST


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
    store.write(prices, tmp_path / "etfs" / "1d", MANIFEST | {"universe": "etfs"})
    macro = pl.DataFrame(
        [(t, f, float(levels[i, k])) for i, t in enumerate(ts) for k, f in enumerate(FACTORS)],
        schema=["timestamp", "factor", "value"],
        orient="row",
    )
    dest = tmp_path / "macro"
    meta = MANIFEST | dict(
        source="synthetic integration fixture",
        availability_method="synthetic bar-end release",
        known_at_verified=True,
        units=UNITS,
    )
    store.write(macro, dest, meta, file="macro.parquet")
    graph = yaml.safe_load(
        (Path(__file__).parents[2] / "config/graphs/macro_sensitivity.yaml").read_text()
    )
    graph.update(name="synthetic_macro", universe="etfs")
    graph["params"].update(window_bars=10, ridge=1.0)
    graph["features"] = {"min_history_bars": 10, "list": [{"name": "momentum", "bars": 3}]}
    graph["sources"]["macro"]["path"] = "macro"
    graph["model"].update(hidden=4, layers=1, heads=1, dropout=0.0)
    graph["baseline"].update(hidden=4, layers=1, dropout=0.0)
    cfg = dict(
        data_root=str(tmp_path),
        data={"bar": "1d"},
        graph=graph,
        normalize="rank",
        horizon_bars=2,
        walk_forward=dict(
            window="rolling",
            train_bars=20,
            val_bars=8,
            test_bars=10,
            holdout_start=date(2018, 4, 1),
        ),
        train=dict(loss="listmle", lr=0.001, batch_size=8, max_epochs=2, patience=2, seed=7),
        eval=dict(quantiles=3, top_k=1, benchmark="SPY", rolling_dates=3, permutation_repeats=1),
        device="cpu",
    )
    return cfg, prices, ts, dest


def builder(fixture, **params):
    cfg, prices, ts, _ = fixture
    return MacroGraphBuilder(
        {
            "prices": prices,
            "features": features.compute(prices, cfg["graph"]["features"]),
            "macro": load_macro(Path(cfg["data_root"]), cfg["graph"]["sources"]["macro"])[0],
        },
        pl.Series(ts),
        **(cfg["graph"]["params"] | params),
    )


def test_layout_mask_baseline_information_and_model_backward(fixture):
    cfg, prices, _, _ = fixture
    snapshots = build_snapshots(cfg).snapshots
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


@pytest.mark.parametrize("scale", ["raw", "percent"])
@pytest.mark.parametrize("mode", ["magnitude", "signed", "constant", "shuffled"])
def test_future_deletion_invariance(fixture, mode, scale):
    b = builder(fixture, edge_mode=mode, beta_scale=scale)
    t = fixture[2][40]
    full = b.build(t)
    trimmed = {name: df.filter(pl.col("timestamp") <= t) for name, df in b._inputs.items()}
    short = MacroGraphBuilder(
        trimmed,
        b._axis.filter(b._axis <= t),
        **(fixture[0]["graph"]["params"] | {"edge_mode": mode, "beta_scale": scale}),
    ).build(t)
    assert np.array_equal(full.x, short.x)
    assert np.array_equal(full.edge_attr, short.edge_attr)


def _change(s, factor):
    return s.x[s.target_mask, s.feature_names.tolist().index(f"macro_change_{factor}")]


def test_macro_not_seen_before_release(fixture):
    b = builder(fixture)
    t = fixture[2][40]
    # A release an hour after t is not known at t: t carries the previous value forward
    # (zero change), and the release enters at the next bar.
    m = b._inputs["macro"]
    mask = (pl.col("timestamp") == t) & (pl.col("factor") == "credit")
    b._inputs["macro"] = m.with_columns(
        pl.when(mask)
        .then(pl.col("timestamp") + timedelta(hours=1))
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    ).sort("timestamp")
    s = b.build(t)
    assert s.target_mask.any() and (_change(s, "credit") == 0).all()
    assert (_change(builder(fixture).build(t), "credit") != 0).all()


def test_macro_carried_forward_however_old_until_a_factor_has_no_spread(fixture):
    ts, t = fixture[2], fixture[2][40]

    def without_credit(days):
        b = builder(fixture)
        m = b._inputs["macro"]
        b._inputs["macro"] = m.filter(
            ~((pl.col("factor") == "credit") & pl.col("timestamp").is_in(days))
        )
        return b.build(t)

    # A 4-bar gap ending at t: the last value is carried, so t's change is zero.
    s = without_credit(ts[37:41])
    assert s.target_mask.sum() == 3 and (_change(s, "credit") == 0).all()
    # Carried through the whole regression window, credit has no spread: no snapshot.
    assert not without_credit(ts[29:41]).target_mask.any()


@pytest.mark.parametrize("scale", ["raw", "percent"])
def test_constant_edges_are_the_snapshots_mean_edge_in_both_channels(fixture, scale):
    t = fixture[2][40]
    real = builder(fixture, beta_scale=scale).build(t)
    s = builder(fixture, edge_mode="constant", beta_scale=scale).build(t)
    assert s.edge_weight == pytest.approx(np.full(15, real.edge_weight.mean()), rel=1e-6)
    assert np.array_equal(s.edge_attr[:, 0], s.edge_weight)
    assert len(set(s.edge_weight.tolist())) == 1  # no per-edge beta information
    assert np.array_equal(s.x, real.x)  # node exposures unchanged


def test_percent_scales_edges_and_interactions_only(fixture):
    t = fixture[2][40]
    for mode in ("magnitude", "signed", "shuffled", "constant"):
        raw = builder(fixture, edge_mode=mode).build(t)
        pct = builder(fixture, edge_mode=mode, beta_scale="percent").build(t)
        assert pct.edge_weight == pytest.approx(100 * raw.edge_weight, rel=1e-6)
        assert pct.edge_attr == pytest.approx(100 * raw.edge_attr, rel=1e-6)
        names = raw.feature_names.tolist()
        inter = [names.index(f"interaction_{f}") for f in FACTORS]
        assert pct.x[:, inter] == pytest.approx(100 * raw.x[:, inter], rel=1e-6)
        assert np.abs(raw.x[:3, inter]).max() > 0
        # Every other column, including the ranked beta_* / magnitude_*, is unchanged.
        rest = [i for i in range(len(names)) if i not in inter]
        assert np.array_equal(pct.x[:, rest], raw.x[:, rest])
    with pytest.raises(ValueError, match="beta_scale"):
        builder(fixture, beta_scale="bps")


def test_signed_edges_preserve_sign(fixture):
    t = fixture[2][40]
    s = builder(fixture, edge_mode="signed").build(t)
    idx = [s.feature_names.tolist().index(f"beta_{f}") for f in FACTORS]
    assert np.array_equal(s.edge_attr[:, 0], s.x[:3, idx].reshape(-1))
    assert np.array_equal(s.edge_weight, np.abs(s.edge_attr[:, 0]))


def test_missing_equity_price_never_forward_filled(fixture):
    ts, t = fixture[2], fixture[2][40]
    bbb = pl.col("ticker") == "BBB"

    def build(prices):
        b = builder(fixture)
        b._inputs["prices"] = prices(b._inputs["prices"])
        s = b.build(t)
        return s.edge_weight[s.edge_index[1] == s.node_ids.tolist().index("BBB")]

    gap = build(lambda p: p.filter(~(bbb & (pl.col("timestamp") == ts[37]))))
    # The missing close's two returns leave the fit; carrying the close would differ.
    prior = fixture[1].filter(bbb & (pl.col("timestamp") == ts[36]))["close"].item()
    filled = build(
        lambda p: p.with_columns(
            close=pl.when(bbb & (pl.col("timestamp") == ts[37]))
            .then(prior)
            .otherwise(pl.col("close"))
        )
    )
    assert not np.allclose(gap, filled)
    # Fewer complete rows than the five factors: the equity is left out.
    b = builder(fixture)
    b._inputs["prices"] = b._inputs["prices"].filter(~(bbb & pl.col("timestamp").is_in(ts[32:39])))
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
    _, _, _, dest = fixture
    lf, meta = store.open(dest)
    original = lf.collect()
    meta = {k: v for k, v in meta.items() if k not in ("file", "rows", "sha256")}
    naive = original.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    store.write(naive, dest, meta, file="macro.parquet")
    with pytest.raises(ValueError, match="UTC offset"):
        load_macro(dest.parent, {"path": dest.name})
    store.write(pl.concat([original, original.head(1)]), dest, meta, file="macro.parquet")
    with pytest.raises(ValueError, match="duplicate"):
        load_macro(dest.parent, {"path": dest.name})


def test_requires_macro_source_and_verified_provenance(fixture):
    cfg, _, _, dest = fixture
    cfg = deepcopy(cfg)
    del cfg["graph"]["sources"]["macro"]
    cfg["data_root"] = str(dest / "nothing_built_here")  # fails before any data is read
    with pytest.raises(ValueError, match="needs sources.macro"):
        build_snapshots(cfg)
    path = dest / "manifest.json"
    meta = json.loads(path.read_text())
    meta["known_at_verified"] = False
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="known-at"):
        load_macro(dest.parent, {"path": dest.name})
    meta |= {"known_at_verified": True, "units": {**UNITS, "rates": "percent"}}
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="units"):
        load_macro(dest.parent, {"path": dest.name})


def test_levels_only_for_level_factors_and_inverted_factor_flips_sign(fixture):
    t = fixture[2][40]
    s = builder(fixture).build(t)  # graph file: invert rates, levels for oil and gold
    names = s.feature_names.tolist()
    assert [n for n in names if n.startswith("macro_level_")] == [
        "macro_level_oil",
        "macro_level_gold",
    ]
    ctx = names.index("context_level")
    ids = s.node_ids.tolist()
    for f in FACTORS:
        row = s.x[ids.index(f"macro:{f}"), ctx]
        assert (row != 0) if f in ("oil", "gold") else (row == 0)
    plain = builder(fixture, invert_factors=[]).build(t)
    for kind in ("macro_change", "beta"):
        k = names.index(f"{kind}_rates")
        assert s.x[:3, k] == pytest.approx(-plain.x[:3, k], abs=1e-6)
        k = names.index(f"{kind}_credit")
        assert s.x[:3, k] == pytest.approx(plain.x[:3, k], abs=1e-6)
    every = builder(fixture, level_factors=list(FACTORS)).build(t)
    assert sum(n.startswith("macro_level_") for n in every.feature_names) == 5
    with pytest.raises(ValueError, match="level_factors"):
        builder(fixture, level_factors=["copper"])


def test_builder_scaled_columns_reach_the_model_unranked(fixture):
    cfg, prices, _, dest = fixture
    snapshots, manifests, _ = build_snapshots(cfg)
    assert manifests["macro"] == json.loads((dest / "manifest.json").read_text())
    s = snapshots[0]
    names = s.feature_names.tolist()
    prescaled = {n for n, p in zip(names, s.feature_prescaled, strict=True) if p}
    assert prescaled == {n for n in names if n.startswith(PRESCALED)}
    assert {"momentum_3", "beta_rates", "magnitude_rates"}.isdisjoint(prescaled)
    data = build_dataset(snapshots[:1], forward_returns(prices, 2), s.timestamp, s.timestamp, cfg)
    cols, x = model_columns(cfg["graph"])[0], data[0]["x"].numpy()
    for j, c in enumerate(cols):
        if c in prescaled:  # as the builder scaled them, although normalize is rank
            assert x[:, j] == pytest.approx(s.x[:, names.index(c)])
    # The rest are ranked across the 3 equities as usual.
    assert sorted(x[:3, cols.index("momentum_3")]) == pytest.approx([-1, 0, 1])


def test_shared_run_logs_macro_provenance_predictions_and_reports(fixture, monkeypatch, tmp_path):
    import mlflow
    from mlflow.artifacts import download_artifacts

    import gnns.config as config
    import gnns.graphs.build as build
    import gnns.train.run as runner
    from gnns.eval.report import report

    cfg, _, _, _ = fixture
    for module in (config, build, runner):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "git_sha", lambda: "synthetic-fixture")
    run_id = runner.run(cfg)
    client = mlflow.MlflowClient()
    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert {
        "predictions_gatv2.parquet",
        "predictions_mlp.parquet",
        "manifest_prices.json",
        "manifest_macro.json",
        "config.yaml",
    } <= artifacts
    predictions = pl.read_parquet(
        download_artifacts(
            run_id=run_id,
            artifact_path="predictions_gatv2.parquet",
            dst_path=str(tmp_path / "readback"),
        )
    )
    last_prediction = predictions["timestamp"].max()
    assert isinstance(last_prediction, datetime)
    assert last_prediction.date() < cfg["walk_forward"]["holdout_start"]
    assert not any(n.startswith("macro:") for n in predictions["ticker"])
    charts = report(run_id, cfg["data_root"])
    assert len(charts) == 9
    assert {"8_score_spread.png", "9_feature_importance.png", "11_feature_correlation.png"} <= set(
        charts
    )
