"""`gnns train`: train the construction's model and matched baseline on every search fold,
score each fold's test period, and log config, metrics and predictions to MLflow.
Charts are made on demand by `gnns report <run_id>` (`gnns.eval.report`)."""

import json
import logging
import tempfile
from pathlib import Path

import mlflow
import polars as pl
import torch
import yaml

from gnns.config import ROOT, git_sha, mlflow_tracking_uri
from gnns.data import store
from gnns.data.labels import forward_returns
from gnns.eval import metrics
from gnns.graphs.build import build_snapshots
from gnns.train.dataset import build_dataset
from gnns.train.losses import LOSSES
from gnns.train.models import build_model
from gnns.train.trainer import predict, train_fold
from gnns.train.walk_forward import walk_forward

log = logging.getLogger(__name__)
ROLES = ("model", "baseline")  # keys in the graph file


def run(cfg: dict) -> str:
    """Returns the MLflow run id. Held-out folds are never touched."""
    graph, train_cfg, h = cfg["graph"], cfg["train"], cfg["horizon_bars"]
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    root = ROOT / cfg["data_root"]

    snaps = build_snapshots(cfg)
    prices = store.load(root, graph["universe"], cfg["data"]["bar"]).collect()
    labels = forward_returns(prices, h)
    axis = prices["timestamp"].unique().sort()
    folds = [
        f
        for f in walk_forward(axis.filter(axis >= snaps[0].timestamp), cfg["walk_forward"], h)
        if not f.holdout
    ]
    data = [
        {
            p: build_dataset(snaps, labels, getattr(f, f"{p}_start"), getattr(f, f"{p}_end"), cfg)
            for p in ("train", "val", "test")
        }
        for f in folds
    ]

    loss_fn = LOSSES[train_cfg["loss"]]
    preds, histories = {}, {}
    for role in ROLES:
        label = graph[role]["name"]
        fold_preds, histories[label] = [], []
        for i, d in enumerate(data, start=1):
            torch.manual_seed(train_cfg["seed"])
            model = build_model(graph[role], d["train"][0]).to(device)
            model, hist = train_fold(model, d["train"], d["val"], train_cfg, loss_fn, device)
            fold_preds.append(predict(model, d["test"], device).with_columns(fold=pl.lit(i)))
            histories[label].append(hist)
            log.info(
                "%s fold %d: %d epochs, best val IC %.3f",
                label,
                i,
                len(hist),
                max(e["val_ic"] for e in hist),
            )
        preds[label] = pl.concat(fold_preds)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        per_year = metrics.bars_per_year(cfg["data"]["bar"]) / h  # rebalances per year
        summary = {
            label: _summarize(p, per_year) | {"val_ic": _val_ic(histories[label])}
            for label, p in preds.items()
        }
        for label, p in preds.items():
            p.write_parquet(out / f"predictions_{label}.parquet")
        (out / "histories.json").write_text(json.dumps(histories))
        return _log_mlflow(cfg, summary, out, root)


def _summarize(pred: pl.DataFrame, periods_per_year: float) -> dict:
    """IC summary plus mean test IC per fold."""
    ic = metrics.daily_ic(pred)
    by_fold = (
        ic.join(pred.select("timestamp", "fold").unique(), on="timestamp")
        .group_by("fold")
        .agg(pl.col("ic").mean())
        .sort("fold")
    )
    return metrics.ic_summary(ic, periods_per_year) | {"fold_ic": by_fold["ic"].to_list()}


def _val_ic(fold_histories: list[list[dict]]) -> float:
    """Mean over folds of the kept (best-epoch) validation IC: the sweep selection metric.
    Optimistic, since the epoch was chosen on it; compare runs with it, don't report it."""
    return sum(max(e["val_ic"] for e in h) for h in fold_histories) / len(fold_histories)


def _log_mlflow(cfg: dict, summary: dict, out: Path, root: Path) -> str:
    mlflow.set_tracking_uri(mlflow_tracking_uri())
    name = cfg["graph"]["name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlruns" / name).as_uri())
    mlflow.set_experiment(name)
    with mlflow.start_run() as active:
        mlflow.set_tag("git_sha", git_sha())
        mlflow.log_params(_flatten({k: v for k, v in cfg.items() if k != "data_root"}))
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        manifest = root / cfg["graph"]["universe"] / cfg["data"]["bar"] / "manifest.json"
        (out / "manifest.json").write_text(manifest.read_text())
        for label, s in summary.items():
            mlflow.log_metrics({f"{label}_{k}": v for k, v in s.items() if k != "fold_ic"})
            for fold, ic in enumerate(s["fold_ic"], start=1):
                mlflow.log_metric(f"{label}_fold_ic", ic, step=fold)
        mlflow.log_artifacts(str(out))
        return active.info.run_id


def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        flat.update(_flatten(v, f"{key}.") if isinstance(v, dict) else {key: str(v)})
    return flat
