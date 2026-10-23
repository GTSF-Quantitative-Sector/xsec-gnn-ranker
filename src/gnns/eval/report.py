"""`gnns report <run_id>`: make the evaluation charts for one MLflow run and attach them to it.

Reads the run's own artifacts (predictions, histories, importance, config), plus the
price data for the feature-correlation chart. It never retrains. Commands that produce a
run call it afterwards (`gnns.cli`); the portfolio charts come from `gnns backtest`.
"""

import json
import logging
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import yaml
from mlflow.artifacts import download_artifacts

from gnns.config import ROOT, mlflow_tracking_uri
from gnns.data import features
from gnns.data.features import column_names
from gnns.eval import metrics, plots

log = logging.getLogger(__name__)


def report(run_id: str, data_root: str) -> list[str]:
    """Download the run's artifacts, make the charts, log them to the same run."""
    mlflow.set_tracking_uri(mlflow_tracking_uri())
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(download_artifacts(run_id=run_id, dst_path=tmp))
        cfg = yaml.safe_load((src / "config.yaml").read_text())
        files = sorted(src.glob("predictions_*.parquet"))
        preds = {f.stem.removeprefix("predictions_"): pl.read_parquet(f) for f in files}
        histories = json.loads((src / "histories.json").read_text())
        imp_path = src / "importance.json"  # absent for runs trained before it existed
        importances = json.loads(imp_path.read_text()) if imp_path.exists() else None
        out = Path(tmp) / "charts"
        out.mkdir()
        make_charts(preds, histories, cfg, out, importances, _feature_corr(cfg, data_root))
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(str(out))
        return sorted(p.name for p in out.iterdir())


def make_charts(
    preds: dict[str, pl.DataFrame],
    histories: dict[str, list[list[dict]]],
    cfg: dict,
    out: Path,
    importances: dict[str, list[dict[str, float]]] | None = None,
    feature_corr: tuple[np.ndarray, list[str]] | None = None,
) -> None:
    """Write the PNGs for one run. `preds` maps model label -> predictions table."""
    eval_cfg = cfg["eval"]
    ics = {label: metrics.daily_ic(p) for label, p in preds.items()}
    qrets = {
        label: metrics.quantile_returns(p, eval_cfg["quantiles"]) for label, p in preds.items()
    }
    fold_ics = {label: _fold_ic(ics[label], p) for label, p in preds.items()}
    first_label = next(iter(preds))

    plots.ic_over_time(ics, eval_cfg["rolling_dates"], out / "1_ic_over_time.png")
    plots.cumulative_ic(ics, out / "3_cumulative_ic.png")
    plots.ic_by_fold(fold_ics, out / "4_ic_by_fold.png")
    plots.quantile_cascade(qrets, out / "5_quantile_cascade.png")
    plots.quantile_curves(qrets[first_label], first_label, out / "6_quantile_curves.png")
    # 7 was the no-cost top-k preview, replaced by gnns backtest's backtest/1_equity.png
    plots.score_spread(preds, out / "8_score_spread.png")
    if importances is not None:
        plots.feature_importance(importances, out / "9_feature_importance.png")
    plots.training_curves(histories, out / "10_training_curves.png")
    if feature_corr is not None:
        plots.feature_correlation(*feature_corr, out / "11_feature_correlation.png")


def _fold_ic(ic: pl.DataFrame, pred: pl.DataFrame) -> pl.DataFrame:
    """(fold, mean_ic, test_year) from a run's per-date IC and its predictions."""
    folds = pred.group_by("fold").agg(test_year=pl.col("timestamp").min().dt.year().cast(pl.String))
    return (
        ic.join(pred.select("timestamp", "fold").unique(), on="timestamp")
        .group_by("fold")
        .agg(mean_ic=pl.col("ic").mean())
        .join(folds, on="fold")
        .sort("fold")
    )


def _feature_corr(cfg: dict, data_root: str) -> tuple[np.ndarray, list[str]] | None:
    """Within-date correlation of the construction's features, search period only."""
    graph = cfg["graph"]
    path = ROOT / data_root / graph["universe"] / cfg["data"]["bar"] / "prices.parquet"
    if not path.exists():
        log.warning("%s dataset not built; skipping feature correlation", graph["universe"])
        return None
    names = column_names(graph["features"]["list"])
    feats = features.compute(pl.read_parquet(path), graph["features"]).filter(
        pl.col("timestamp").dt.date() < cfg["walk_forward"]["holdout_start"]
    )
    return metrics.cross_sectional_corr(feats, names), names
