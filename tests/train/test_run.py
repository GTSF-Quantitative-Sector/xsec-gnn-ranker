import json
from pathlib import Path

from mlflow.artifacts import download_artifacts

from gnns.graphs.build import build_snapshots
from gnns.train import run
from tests.graphs.test_build import _toy_cfg


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
