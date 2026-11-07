import json
from pathlib import Path

import pytest
import torch
from mlflow.artifacts import download_artifacts
from torch_geometric.data import Data

from gnns.graphs.build import build_snapshots
from gnns.train import run
from gnns.train.models import build_model
from tests.graphs.test_build import _toy_cfg

TRAIN = {"loss": "listmle", "lr": 0.01, "batch_size": 16, "seed": 0}


def _roles(model_train=None, baseline_train=None) -> dict:
    model, baseline = {"name": "gatv2"}, {"name": "mlp"}
    if model_train is not None:
        model["train"] = model_train
    if baseline_train is not None:
        baseline["train"] = baseline_train
    return {"train": TRAIN, "graph": {"model": model, "baseline": baseline}}


def test_each_model_overrides_the_shared_training_defaults():
    cfg = _roles(model_train={"lr": 0.001}, baseline_train={})
    assert run.train_settings(cfg, "model") == TRAIN | {"lr": 0.001}
    assert run.train_settings(cfg, "baseline") == TRAIN
    assert run.train_settings(_roles(), "model") == TRAIN  # no train: block at all


def test_seed_and_unknown_keys_in_a_model_train_block_raise():
    with pytest.raises(ValueError, match="seed is shared by both models"):
        run.train_settings(_roles(model_train={"seed": 1}), "model")
    with pytest.raises(ValueError, match=r"keys \['momentum'\] are not training settings"):
        run.train_settings(_roles(baseline_train={"momentum": 1}), "baseline")


def test_build_model_ignores_the_train_block():
    sample = Data(x=torch.zeros(3, 2), edge_attr=torch.zeros(0, 0))
    model = build_model({"name": "mlp", "hidden": 4, "layers": 1, "train": {"lr": 1.0}}, sample)
    assert model.head.in_features == 4


def test_run_logs_one_manifest_per_dataset_as_opened(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(run, "mlflow_tracking_uri", lambda: uri)
    monkeypatch.setattr(run, "ROOT", tmp_path)  # experiment artifacts go to tmp_path/mlruns
    cfg = _toy_cfg(tmp_path, monkeypatch)
    _, manifests = build_snapshots(cfg)
    # Rebuilding a dataset after it was opened must not change what the run logs.
    (tmp_path / "events" / "manifest.json").write_text("{}")

    out = tmp_path / "out"
    out.mkdir()
    run_id = run._log_mlflow(cfg, {"m": {"mean_ic": 0.1, "fold_ic": [0.1]}}, out, manifests)

    got = Path(download_artifacts(run_id=run_id, dst_path=str(tmp_path / "dl")))
    logged = {p.stem.removeprefix("manifest_"): p for p in got.glob("manifest_*.json")}
    assert logged.keys() == {"prices", "events"}
    for name, path in logged.items():
        assert json.loads(path.read_text()) == manifests[name]
