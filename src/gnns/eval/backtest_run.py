"""`gnns backtest <run_id>`: backtest one MLflow run's predictions and attach the results.

Horizon and bar come from the run's saved config (the predictions depend on them); the
portfolio rules come from the current `backtest:` block, so an old run can be re-costed
without retraining. Reruns replace the metrics, `backtest.*` tags and `backtest/`
artifacts. Commands that produce a run call it afterwards (`gnns.cli`).
"""

import tempfile
from pathlib import Path

import mlflow
import polars as pl
import yaml
from mlflow.artifacts import download_artifacts

from gnns.config import ROOT, mlflow_tracking_uri
from gnns.data.labels import forward_returns
from gnns.eval import backtest as bt
from gnns.eval import metrics, plots

REFS = ("benchmark", "risk_free")  # keys in `backtest:`, each a benchmarks-universe ticker


class BenchmarksMissing(Exception):
    """The benchmarks dataset, or the benchmark / risk-free ticker in it, is not built."""


def backtest(run_id: str, cfg: dict) -> dict[str, float]:
    """Backtest every model in the run with the current `backtest:` settings; log to it."""
    mlflow.set_tracking_uri(mlflow_tracking_uri())
    settings = cfg["backtest"]
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(download_artifacts(run_id=run_id, dst_path=str(Path(tmp) / "run")))
        run_cfg = yaml.safe_load((src / "config.yaml").read_text())
        files = sorted(src.glob("predictions_*.parquet"))
        preds = {f.stem.removeprefix("predictions_"): pl.read_parquet(f) for f in files}
        refs = reference_returns(run_cfg, settings, cfg["data_root"])
        per_year = metrics.bars_per_year(run_cfg["data"]["bar"]) / run_cfg["horizon_bars"]
        out = Path(tmp) / "out"
        out.mkdir()
        results = run_backtest(preds, refs, settings, per_year, out)
        (out / "backtest.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(results)
            mlflow.set_tags({f"backtest.{k}": str(v) for k, v in settings.items()})
            mlflow.log_artifacts(str(out), "backtest")
    return results


def run_backtest(
    preds: dict[str, pl.DataFrame],
    refs: pl.DataFrame,
    settings: dict,
    per_year: float,
    out: Path,
) -> dict[str, float]:
    """Simulate each model, write per-period tables and charts to `out`, and return
    metrics keyed `<model>_<metric>`."""
    universe = metrics.universe_returns(next(iter(preds.values()))).rename({"ret": "universe"})
    periods, results = {}, {}
    for label, pred in preds.items():
        try:
            p = bt.simulate(pred, settings["top_k"], settings["weighting"])
        except ValueError as e:
            e.add_note(f"model: {label}")
            raise
        missing = p.join(refs, on="timestamp", how="anti")
        if missing.height:
            raise ValueError(
                f"{missing.height} rebalance dates have no {settings['benchmark']} or "
                f"{settings['risk_free']} return, first {missing['timestamp'][0]}"
            )
        p = p.join(refs, on="timestamp").join(universe, on="timestamp")
        p.write_parquet(out / f"periods_{label}.parquet")
        periods[label] = p
        summary = bt.summarize(p, settings["cost_bps"], per_year)
        results |= {f"{label}_{k}": v for k, v in summary.items()}
    years = {
        f"{label} {bps:g} bps": bt.yearly(p, bps, per_year)
        for label, p in periods.items()
        for bps in settings["cost_bps"]
    }
    plots.backtest_equity(
        periods, settings["cost_bps"], settings["benchmark"], out / "1_equity.png"
    )
    plots.backtest_drawdown(periods, settings["cost_bps"], out / "2_drawdown.png")
    plots.backtest_per_year(years, out / "3_per_year.png")
    return results


def reference_returns(run_cfg: dict, settings: dict, data_root: str) -> pl.DataFrame:
    """(timestamp, benchmark, risk_free) forward returns over the run's horizon."""
    path = ROOT / data_root / "benchmarks" / run_cfg["data"]["bar"] / "prices.parquet"
    if not path.exists():
        raise BenchmarksMissing(f"{path} not built; run gnns build-data with benchmarks")
    fwd = forward_returns(pl.read_parquet(path), run_cfg["horizon_bars"])
    tickers = set(fwd["ticker"].unique().to_list())
    cols = []
    for role in REFS:
        ticker = settings[role]
        if ticker not in tickers:
            raise BenchmarksMissing(
                f"backtest.{role} {ticker} is not in the benchmarks dataset; add it to "
                "config/universes/benchmarks.txt and run gnns build-data"
            )
        rows = fwd.filter(pl.col("ticker") == ticker)
        cols.append(rows.select("timestamp", pl.col("fwd_return").alias(role)))
    return cols[0].join(cols[1], on="timestamp")


def summary_lines(results: dict[str, float], cost_bps: list[float]) -> list[str]:
    """One terminal line per model and cost level, then turnover and break-even cost."""
    suffix = "_break_even_bps"
    labels = [k.removesuffix(suffix) for k in results if k.endswith(suffix)]
    lines = []
    for label in labels:
        for bps in cost_bps:
            m = {
                k: results[f"{label}_{k}_{bps:g}bps"]
                for k in ("ann_return", "sharpe", "ir", "max_drawdown")
            }
            lines.append(
                f"  {label} {bps:g} bps: return {m['ann_return']:.1%}/yr, "
                f"Sharpe {m['sharpe']:.2f}, IR {m['ir']:.2f}, "
                f"max drawdown {m['max_drawdown']:.1%}"
            )
        lines.append(
            f"  {label}: turnover {results[f'{label}_turnover_annual']:.1f}x/yr, "
            f"break-even cost {results[f'{label}{suffix}']:.0f} bps"
        )
    return lines
