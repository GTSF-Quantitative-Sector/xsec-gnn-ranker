"""`gnns report <run_id>`: make the evaluation charts for one MLflow run and attach them to it.

Reads only the run's own artifacts (predictions, histories, config), so it never retrains.
"""

import json
import logging
import tempfile
from pathlib import Path

import mlflow
import polars as pl
import yaml
from mlflow.artifacts import download_artifacts

from gnns.config import ROOT, mlflow_tracking_uri
from gnns.data.labels import forward_returns
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
        out = Path(tmp) / "charts"
        out.mkdir()
        make_charts(preds, histories, cfg, _benchmark(cfg, data_root), out)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(str(out))
        return sorted(p.name for p in out.iterdir())


def make_charts(
    preds: dict[str, pl.DataFrame],
    histories: dict[str, list[list[dict]]],
    cfg: dict,
    benchmark: pl.DataFrame | None,
    out: Path,
) -> None:
    """Write the PNGs for one run. `preds` maps model label -> predictions table."""
    eval_cfg = cfg["eval"]
    ics = {label: metrics.daily_ic(p) for label, p in preds.items()}
    qrets = {
        label: metrics.quantile_returns(p, eval_cfg["quantiles"]) for label, p in preds.items()
    }
    fold_ics = {label: _fold_ic(ics[label], p) for label, p in preds.items()}
    curves = {
        f"{label} top-{eval_cfg['top_k']}": metrics.top_k_returns(p, eval_cfg["top_k"])
        for label, p in preds.items()
    }
    first_label, first = next(iter(preds.items()))
    curves["equal-weight universe"] = metrics.universe_returns(first)
    if benchmark is not None:
        curves[eval_cfg["benchmark"]] = benchmark.join(
            first.select("timestamp").unique(), on="timestamp"
        ).sort("timestamp")

    plots.ic_over_time(ics, eval_cfg["rolling_dates"], out / "1_ic_over_time.png")
    plots.cumulative_ic(ics, out / "3_cumulative_ic.png")
    plots.ic_by_fold(fold_ics, out / "4_ic_by_fold.png")
    plots.quantile_cascade(qrets, out / "5_quantile_cascade.png")
    plots.quantile_curves(qrets[first_label], first_label, out / "6_quantile_curves.png")
    plots.equity_curves(curves, out / "7_equity_curves.png")
    plots.training_curves(histories, out / "10_training_curves.png")


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


def _benchmark(cfg: dict, data_root: str) -> pl.DataFrame | None:
    """(timestamp, ret) forward returns of the benchmark ticker, or None if not built."""
    path = ROOT / data_root / "benchmarks" / cfg["data"]["bar"] / "prices.parquet"
    if not path.exists():
        log.warning("benchmarks dataset not built; skipping %s", cfg["eval"]["benchmark"])
        return None
    bench = forward_returns(pl.read_parquet(path), cfg["horizon_bars"])
    return bench.filter(pl.col("ticker") == cfg["eval"]["benchmark"]).select(
        "timestamp", ret="fwd_return"
    )
