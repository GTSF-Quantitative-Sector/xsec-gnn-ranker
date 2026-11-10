from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import mlflow
import numpy as np
import polars as pl
import pytest
import yaml
from mlflow.artifacts import download_artifacts

from gnns.data import store
from gnns.train import run, sweep
from tests.data.test_store import MANIFEST

T0 = datetime(2024, 1, 1, 16, tzinfo=ZoneInfo("America/New_York"))
SWEEP = {
    "seeds": [0, 1],
    "model_grid": {"params.window_bars": [5, 15], "model.hidden": [4, 8]},
    "baseline_grid": {"baseline.hidden": [4, 8]},
}


def toy_cfg(tmp_path, monkeypatch, sweep_block: dict | None = None) -> dict:
    """4 ETFs x 100 daily bars, horizon 1, small folds and models: trains in seconds.
    MLflow and experiment artifacts go to tmp_path."""
    rng = np.random.default_rng(0)
    rows = [
        (T0 + timedelta(days=i), k, float(c), float(c), float(c), float(c), 100.0)
        for k in ("XLB", "XLE", "XLF", "XLK")
        for i, c in enumerate(100 * np.cumprod(1 + rng.normal(0, 0.02, 100)))
    ]
    prices = pl.DataFrame(rows, schema=store.PRICE_COLS, orient="row").sort("timestamp", "ticker")
    store.write(prices, tmp_path / "etfs" / "1d", MANIFEST | {"universe": "etfs"})
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(run, "mlflow_tracking_uri", lambda: uri)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    return {
        "data_root": str(tmp_path),
        "data": {"bar": "1d"},
        "horizon_bars": 1,
        "normalize": "rank",
        "device": "cpu",
        "walk_forward": {
            "window": "rolling",
            "train_bars": 30,
            "val_bars": 10,
            "test_bars": 10,
            "holdout_start": (T0 + timedelta(days=80)).date(),
        },
        "train": {
            "loss": "listmle",
            "lr": 0.01,
            "batch_size": 8,
            "max_epochs": 2,
            "patience": 1,
            "seed": 0,
        },
        "eval": {"permutation_repeats": 1},
        "graph": {
            "name": "toy",
            "builder": "sector_etf_correlation",
            "universe": "etfs",
            "params": {"window_bars": 5},
            "features": {
                "min_history_bars": 5,
                "list": [{"name": "momentum", "bars": 5}, {"name": "volatility", "bars": 5}],
            },
            "extra_model_features": [],
            "model": {"name": "gatv2", "hidden": 4, "layers": 1, "heads": 1, "dropout": 0.0},
            "model_edge_attrs": [],
            "baseline": {"name": "mlp", "hidden": 4, "layers": 1, "dropout": 0.0},
            "sweep": SWEEP if sweep_block is None else sweep_block,
        },
    }


def _spec_cfg(grid: str, key: str) -> dict:
    """A config whose only grid entry under `grid` is `key` (no data needed)."""
    graph = {
        "name": "toy",
        "params": {"window_bars": 5},
        "features": {"min_history_bars": 5, "list": []},
        "extra_model_features": [],
        "model": {"name": "gatv2", "hidden": 4},
        "model_edge_attrs": [],
        "baseline": {"name": "mlp", "hidden": 4},
        "builder": "sector_etf_correlation",
    }
    grids = {"model_grid": {"model.hidden": [4]}, "baseline_grid": {"baseline.hidden": [4]}}
    grids[grid] = {key: [1]}
    graph["sweep"] = {"seeds": [0], **grids}
    return {"graph": graph, "train": {"loss": "listmle", "lr": 0.01, "seed": 0}}


@pytest.mark.parametrize(
    ("grid", "key", "match"),
    [
        ("model_grid", "baseline.hidden", "belongs in baseline_grid"),
        ("baseline_grid", "model.hidden", "belongs in model_grid"),
        ("baseline_grid", "params.window_bars", "data side is tuned in stage 1"),
        ("model_grid", "builder", "fixed for the whole experiment"),
        ("model_grid", "horizon_bars", "fixed for the whole experiment"),
        ("model_grid", "model.name", "architecture is fixed"),
        ("model_grid", "features.min_history_bars", "derived, not tuned"),
        ("model_grid", "model.train.seed", "list seeds in sweep.seeds"),
        ("model_grid", "model.train.momentum", "not a training setting"),
        ("model_grid", "params.windw_bars", "no such key"),
        ("model_grid", "params", "sweep keys inside params"),
    ],
)
def test_grid_keys_outside_their_stage_are_rejected(grid, key, match):
    with pytest.raises(ValueError, match=match):
        sweep.parse_spec(_spec_cfg(grid, key))


def test_tunable_keys_are_accepted():
    for grid, key in [
        ("model_grid", "params.window_bars"),
        ("model_grid", "features.list"),
        ("model_grid", "extra_model_features"),
        ("model_grid", "model.train.lr"),
        ("model_grid", "model_edge_attrs"),
        ("baseline_grid", "baseline.train.lr"),
    ]:
        sweep.parse_spec(_spec_cfg(grid, key))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"seeds": []}, "non-empty list of distinct ints"),
        ({"seeds": [0, 0]}, "non-empty list of distinct ints"),
        ({"baseline_grid": {}}, "at least one key"),
        ({"model_grid": {"model.hidden": []}}, "non-empty list of values"),
        ({"extra": 1}, "unknown keys"),
    ],
)
def test_malformed_spec_is_rejected(change, match):
    cfg = _spec_cfg("model_grid", "model.hidden")
    cfg["graph"]["sweep"] |= change
    with pytest.raises(ValueError, match=match):
        sweep.parse_spec(cfg)


def test_missing_sweep_block_or_grid_raises():
    cfg = _spec_cfg("model_grid", "model.hidden")
    del cfg["graph"]["sweep"]["baseline_grid"]
    with pytest.raises(ValueError, match="needs"):
        sweep.parse_spec(cfg)
    del cfg["graph"]["sweep"]
    with pytest.raises(ValueError, match="no sweep: block"):
        sweep.parse_spec(cfg)


def test_expand_is_cartesian_in_grid_order_and_names_list_values_by_position():
    lists = [[{"name": "momentum", "bars": 5}], [{"name": "momentum", "bars": 9}]]
    combos = sweep.expand({"model.hidden": [4, 8], "features.list": lists})
    assert [c.overrides["model.hidden"] for c in combos] == [4, 4, 8, 8]
    assert [c.index for c in combos] == [0, 1, 2, 3]
    assert combos[1].overrides["features.list"] == lists[1]
    assert combos[1].name == "model.hidden=4, features.list=#2"


def test_apply_sets_nested_paths_and_leaves_the_original_alone():
    cfg = _spec_cfg("model_grid", "model.hidden")
    out = sweep.apply(cfg, {"model.train.lr": 0.001, "params.window_bars": 9})
    assert out["graph"]["model"]["train"] == {"lr": 0.001}
    assert out["graph"]["params"]["window_bars"] == 9
    assert "train" not in cfg["graph"]["model"]
    assert cfg["graph"]["params"]["window_bars"] == 5


@pytest.mark.parametrize(
    ("window", "feature_bars", "file_value", "expected"),
    [(5, 5, 5, 5), (15, 5, 5, 15), (5, 20, 5, 20), (5, 5, 30, 30)],
)
def test_required_history_is_the_longest_of_file_feature_and_price_windows(
    window, feature_bars, file_value, expected
):
    cfg = _spec_cfg("model_grid", "model.hidden")
    cfg["graph"]["params"]["window_bars"] = window
    cfg["graph"]["features"] = {
        "min_history_bars": file_value,
        "list": [{"name": "momentum", "bars": feature_bars}],
    }
    assert sweep.required_history(cfg) == expected


def test_better_prefers_higher_score_then_earlier_combination():
    def best(score, index):
        return sweep.Best(sweep.Combo(index, {}, ""), {}, score, {}, None)  # type: ignore[arg-type]

    assert sweep.better(0.1, 3, None)
    assert sweep.better(0.2, 3, best(0.1, 0))
    assert not sweep.better(0.1, 3, best(0.2, 0))
    assert sweep.better(0.1, 1, best(0.1, 2))  # tie: earlier in grid order wins
    assert not sweep.better(0.1, 2, best(0.1, 1))


def test_bad_combination_fails_before_any_training(tmp_path, monkeypatch):
    block = SWEEP | {"model_grid": {"model.train.loss": ["listmle", "nope"]}}
    cfg = toy_cfg(tmp_path, monkeypatch, block)
    monkeypatch.setattr(run, "build", lambda cfg: pytest.fail("built before checking"))
    with pytest.raises(ValueError, match="unknown loss 'nope'"):
        sweep.sweep(cfg)


def test_failed_sweep_marks_the_parent_run_failed(tmp_path, monkeypatch):
    cfg = toy_cfg(tmp_path, monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(run, "train", boom)
    with pytest.raises(RuntimeError, match="boom"):
        sweep.sweep(cfg)
    runs = mlflow.search_runs(experiment_names=["toy"], output_format="list")
    assert [(r.info.run_name, r.info.status) for r in runs] == [("sweep", "FAILED")]


def test_sweep_end_to_end(tmp_path, monkeypatch):
    cfg = toy_cfg(tmp_path, monkeypatch)
    builds = []
    real_build = run.build

    def counting_build(c):
        builds.append(c["graph"]["params"]["window_bars"])
        return real_build(c)

    monkeypatch.setattr(run, "build", counting_build)
    result = sweep.sweep(cfg)

    # One build per distinct data-side setting in stage 1; stage 2 reuses the winner's.
    assert sorted(builds) == [5, 15]
    # Every combination got the history the largest window needs, so the same folds.
    runs = mlflow.search_runs(experiment_names=["toy"], output_format="list")
    by_stage: dict[str, list] = {}
    for r in runs:
        by_stage.setdefault(r.data.tags["sweep.stage"], []).append(r)
    assert {k: len(v) for k, v in by_stage.items()} == {"parent": 1, "1": 8, "2": 4, "final": 2}
    parent = by_stage["parent"][0]
    assert parent.info.status == "FINISHED"
    assert parent.info.run_id == result.parent_run_id
    children = [r for k, v in by_stage.items() if k != "parent" for r in v]
    assert all(r.data.tags["mlflow.parentRunId"] == result.parent_run_id for r in children)

    def artifacts(r):
        return {f.path for f in mlflow.MlflowClient().list_artifacts(r.info.run_id)}

    def config(r):
        path = download_artifacts(run_id=r.info.run_id, artifact_path="config.yaml")
        return yaml.safe_load(Path(path).read_text())

    # Stage runs: validation metrics only, one model, no predictions.
    for stage, label in (("1", "gatv2"), ("2", "mlp")):
        for r in by_stage[stage]:
            assert f"{label}_val_ic" in r.data.metrics
            assert not any(k.endswith("_mean_ic") for k in r.data.metrics)
            assert not any(p.startswith("predictions_") for p in artifacts(r))
            assert config(r)["graph"]["features"]["min_history_bars"] == 15

    # Winners: the highest mean validation IC over seeds in each stage.
    table = pl.read_csv(
        download_artifacts(run_id=result.parent_run_id, artifact_path="results.csv")
    )
    for stage in (1, 2):
        means = (
            table.filter(pl.col("stage") == stage)
            .group_by("combo", "seed")
            .agg(pl.col("val_ic").mean())
            .group_by("combo")
            .agg(pl.col("val_ic").mean())
            .sort("val_ic", "combo", descending=[True, False])
        )
        assert result.summary[f"stage_{stage}"]["winner"] == means["combo"][0]
    assert table.height == (8 + 4) * len(result.summary["val_lift_by_fold_and_seed"]) // 2

    # Final runs: both models, test metrics, importance, predictions; no retraining, so
    # each model's validation IC equals its winning stage run's for the same seed.
    s1, s2 = result.summary["stage_1"], result.summary["stage_2"]
    for r in by_stage["final"]:
        seed = r.data.tags["sweep.seed"]
        names = artifacts(r)
        assert {"predictions_gatv2.parquet", "predictions_mlp.parquet"} <= names
        assert "importance.json" in names
        assert {"gatv2_mean_ic", "mlp_mean_ic", "gatv2_imp_edges"} <= r.data.metrics.keys()
        for stage, label, s in (("1", "gatv2", s1), ("2", "mlp", s2)):
            (winner,) = [
                w
                for w in by_stage[stage]
                if w.data.tags["sweep.combo"] == str(s["winner"])
                and w.data.tags["sweep.seed"] == seed
            ]
            assert r.data.metrics[f"{label}_val_ic"] == winner.data.metrics[f"{label}_val_ic"]
        final = config(r)["graph"]
        assert final["params"]["window_bars"] == s1["overrides"]["params.window_bars"]
        assert final["model"]["hidden"] == s1["overrides"]["model.hidden"]
        assert final["baseline"]["hidden"] == s2["overrides"]["baseline.hidden"]

    # frozen.yaml: both winners' values plus the derived history, nested like the file.
    frozen = yaml.safe_load(
        Path(
            download_artifacts(run_id=result.parent_run_id, artifact_path="frozen.yaml")
        ).read_text()
    )
    expected = {
        "params": {"window_bars": s1["overrides"]["params.window_bars"]},
        "features": {"min_history_bars": 15},
        "model": {"hidden": s1["overrides"]["model.hidden"]},
        "baseline": {"hidden": s2["overrides"]["baseline.hidden"]},
    }
    assert frozen == result.frozen == expected
    assert list(frozen) == list(expected)  # graph-file section order
