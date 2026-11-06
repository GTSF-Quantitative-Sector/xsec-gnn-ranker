"""`gnns train`: train the construction's model and matched baseline on every search fold,
score each fold's test period, and log config, metrics and predictions to MLflow.
Charts are made on demand by `gnns report <run_id>` (`gnns.eval.report`).

Split along its seams so `gnns sweep` (`gnns.train.sweep`) reuses each step:
`build` (snapshots, labels, folds, datasets) -> `train` (one model on every fold) ->
`summarize` (metrics) -> `log_run` (MLflow). `run` calls all four for both models.
"""

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import polars as pl
import torch
import yaml
from torch import nn
from torch_geometric.data import Data

from gnns.config import ROOT, git_sha, mlflow_tracking_uri
from gnns.data import store
from gnns.data.labels import forward_returns
from gnns.eval import metrics
from gnns.eval.importance import permutation_importance
from gnns.graphs.build import build_snapshots
from gnns.train.dataset import build_dataset, model_columns
from gnns.train.losses import LOSSES, no_info_loss
from gnns.train.models import build_model
from gnns.train.trainer import predict, train_fold
from gnns.train.walk_forward import Fold, walk_forward

log = logging.getLogger(__name__)
ROLES = ("model", "baseline")  # keys in the graph file


@dataclass
class Built:
    """What training needs for one data-side setting: the search folds, each fold's
    train / val / test samples, and the manifest of every dataset opened."""

    folds: list[Fold]
    data: list[dict[str, list[Data]]]
    manifests: dict[str, dict]


@dataclass
class Trained:
    """One model (a role: `model` or `baseline`) trained on every fold of a `Built`."""

    role: str
    label: str  # the model's name, e.g. gatv2; keys its metrics and artifacts
    predictions: pl.DataFrame  # test periods, with a `fold` column
    histories: list[list[dict]]  # per fold, per epoch
    models: list[nn.Module]  # per fold, best epoch, on CPU (weights are never saved)
    importances: list[dict[str, float]] | None = None  # per fold, set by `importance`


def run(cfg: dict) -> str:
    """Returns the MLflow run id. Held-out folds are never touched."""
    labels = [cfg["graph"][role]["name"] for role in ROLES]
    if len(set(labels)) < len(labels):
        raise ValueError(f"model and baseline need different names, got {labels}")
    built = build(cfg)
    device = pick_device(cfg)
    results = {}
    for role in ROLES:
        result = train(built, role, cfg, device)
        result.importances = importance(result, built, cfg, device)
        results[result.label] = result
    return log_run(cfg, summarize(results, cfg), results, built.manifests)


def pick_device(cfg: dict) -> str:
    return cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")


def build(cfg: dict) -> Built:
    """Snapshots, labels, search folds and per-fold datasets for the configured graph."""
    graph, h = cfg["graph"], cfg["horizon_bars"]
    snaps, manifests = build_snapshots(cfg)  # manifests as read at open time
    prices = store.load(ROOT / cfg["data_root"], graph["universe"], cfg["data"]["bar"]).collect()
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
    return Built(folds, data, manifests)


def train_settings(cfg: dict, role: str) -> dict:
    """The shared `train:` block with the model's own `train:` overrides on top."""
    own = cfg["graph"][role].get("train") or {}
    if "seed" in own:
        raise ValueError(
            f"graph {role}.train.seed: the seed is shared by both models; "
            "set train.seed in config.yaml (gnns sweep uses sweep.seeds)"
        )
    if unknown := sorted(own.keys() - cfg["train"].keys()):
        raise ValueError(
            f"graph {role}.train keys {unknown} are not training settings; "
            f"valid: {sorted(cfg['train'].keys() - {'seed'})}"
        )
    return cfg["train"] | own


def train(built: Built, role: str, cfg: dict, device: str) -> Trained:
    """Train the graph file's `role` model on every fold and score each test period."""
    spec, settings = cfg["graph"][role], train_settings(cfg, role)
    loss_fn = LOSSES[settings["loss"]]
    preds, histories, models = [], [], []
    for i, d in enumerate(built.data, start=1):
        torch.manual_seed(settings["seed"])
        model = build_model(spec, d["train"][0]).to(device)
        model, hist = train_fold(model, d["train"], d["val"], settings, loss_fn, device)
        ref = no_info_loss(settings["loss"], d["train"])
        histories.append([e | {"no_info_loss": ref} for e in hist])
        preds.append(predict(model, d["test"], device).with_columns(fold=pl.lit(i)))
        models.append(model.cpu())
        log.info(
            "%s fold %d: %d epochs, best val IC %.3f",
            spec["name"],
            i,
            len(hist),
            max(e["val_ic"] for e in hist),
        )
    return Trained(role, spec["name"], pl.concat(preds), histories, models)


def importance(result: Trained, built: Built, cfg: dict, device: str) -> list[dict[str, float]]:
    """Permutation importance of each fold's kept model, on that fold's validation dates."""
    columns, repeats = model_columns(cfg["graph"])[0], cfg["eval"]["permutation_repeats"]
    out = []
    for model, d in zip(result.models, built.data, strict=True):
        out.append(permutation_importance(model.to(device), d["val"], columns, repeats, device))
        model.cpu()
    return out


def summarize(results: dict[str, Trained], cfg: dict, test: bool = True) -> dict[str, dict]:
    """Metrics per model label. Always `val_ic` and `fold_val_ic`; with `test`, the test IC
    summary and `fold_ic`; with importances, `imp_<input>`. Lists are per fold
    (`log_run` steps them by fold)."""
    per_year = metrics.bars_per_year(cfg["data"]["bar"]) / cfg["horizon_bars"]  # rebalances
    out = {}
    for label, r in results.items():
        fold_val = fold_val_ic(r.histories)
        s: dict = {"val_ic": sum(fold_val) / len(fold_val), "fold_val_ic": fold_val}
        if test:
            s |= _test_summary(r.predictions, per_year)
        if r.importances is not None:
            s |= {f"imp_{k}": v for k, v in _mean_importance(r.importances).items()}
        out[label] = s
    return out


def log_run(
    cfg: dict,
    summary: dict[str, dict],
    results: dict[str, Trained],
    manifests: dict[str, dict],
    *,
    predictions: bool = True,
    parent_run_id: str | None = None,
    tags: dict[str, str] | None = None,
    run_name: str | None = None,
) -> str:
    """Log one run: config, git SHA, dataset manifests, the summary's metrics, training
    histories, and (with `predictions`) test predictions and importance. With
    `parent_run_id` the run is that run's child. Returns the run id."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "histories.json").write_text(
            json.dumps({label: r.histories for label, r in results.items()})
        )
        if predictions:
            for label, r in results.items():
                r.predictions.write_parquet(out / f"predictions_{label}.parquet")
        if results and all(r.importances is not None for r in results.values()):
            (out / "importance.json").write_text(
                json.dumps({label: r.importances for label, r in results.items()})
            )
        all_tags = (tags or {}) | ({"mlflow.parentRunId": parent_run_id} if parent_run_id else {})
        return _log_mlflow(cfg, summary, out, manifests, tags=all_tags, run_name=run_name)


def fold_val_ic(histories: list[list[dict]]) -> list[float]:
    """Each fold's kept (best-epoch) validation IC. Their mean is `val_ic`, the sweep
    selection metric. Optimistic, since the epoch was chosen on it; compare runs with it,
    don't report it."""
    return [max(e["val_ic"] for e in h) for h in histories]


def _test_summary(pred: pl.DataFrame, periods_per_year: float) -> dict:
    """Test IC summary plus mean test IC per fold."""
    ic = metrics.daily_ic(pred)
    by_fold = (
        ic.join(pred.select("timestamp", "fold").unique(), on="timestamp")
        .group_by("fold")
        .agg(pl.col("ic").mean())
        .sort("fold")
    )
    return metrics.ic_summary(ic, periods_per_year) | {"fold_ic": by_fold["ic"].to_list()}


def _mean_importance(folds: list[dict[str, float]]) -> dict[str, float]:
    """Average each input's permutation importance over folds."""
    return {k: sum(f[k] for f in folds) / len(folds) for k in folds[0]}


def experiment(name: str) -> str:
    """Set (creating if needed) the construction's MLflow experiment; returns its id."""
    mlflow.set_tracking_uri(mlflow_tracking_uri())
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlruns" / name).as_uri())
    exp = mlflow.set_experiment(name)
    # MLflow 3's UI only shows the runs table for experiments tagged as classic ML.
    mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
    return exp.experiment_id


def _log_mlflow(
    cfg: dict,
    summary: dict,
    out: Path,
    manifests: dict[str, dict],
    *,
    tags: dict[str, str] | None = None,
    run_name: str | None = None,
) -> str:
    """Log config, git SHA, one `manifest_<name>.json` per dataset opened, metrics (list
    values stepped by fold) and `out`."""
    experiment(cfg["graph"]["name"])
    with mlflow.start_run(run_name=run_name, tags=tags) as active:
        mlflow.set_tag("git_sha", git_sha())
        mlflow.log_params(_flatten({k: v for k, v in cfg.items() if k != "data_root"}))
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        for dataset, manifest in manifests.items():
            (out / f"manifest_{dataset}.json").write_text(json.dumps(manifest, indent=2))
        for label, s in summary.items():
            mlflow.log_metrics({f"{label}_{k}": v for k, v in s.items() if not isinstance(v, list)})
            for k, values in ((k, v) for k, v in s.items() if isinstance(v, list)):
                for fold, value in enumerate(values, start=1):
                    mlflow.log_metric(f"{label}_{k}", value, step=fold)
        mlflow.log_artifacts(str(out))
        return active.info.run_id


def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        flat.update(_flatten(v, f"{key}.") if isinstance(v, dict) else {key: str(v)})
    return flat
