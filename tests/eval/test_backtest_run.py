import json
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest
import yaml
from mlflow.artifacts import download_artifacts

from gnns.data import store
from gnns.eval import backtest_run
from gnns.eval.backtest_run import BenchmarksMissing, reference_returns, run_backtest
from tests.data.test_store import MANIFEST

SETTINGS = {
    "top_k": 2,
    "weighting": "equal",
    "cost_bps": [0, 3],
    "benchmark": "SPY",
    "risk_free": "BIL",
}
DATES = [datetime(2020, 1, 3, 16) + timedelta(weeks=w) for w in range(30)]


def _pred(seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [
        (t, k, float(rng.normal()), float(rng.normal(0, 0.02))) for t in DATES for k in "ABCDEF"
    ]
    return pl.DataFrame(rows, schema=["timestamp", "ticker", "score", "fwd_return"], orient="row")


def _refs(dates: list[datetime]) -> pl.DataFrame:
    n = len(dates)
    return pl.DataFrame({"timestamp": dates, "benchmark": [0.002] * n, "risk_free": [0.0005] * n})


def test_run_backtest_writes_tables_charts_and_metrics(tmp_path):
    preds = {"gatv2": _pred(0), "mlp": _pred(1)}
    m = run_backtest(preds, _refs(DATES), SETTINGS, 5, "1d", tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "1_equity.png",
        "2_drawdown.png",
        "3_per_year.png",
        "rebalances_gatv2.parquet",
        "rebalances_mlp.parquet",
    ]
    assert {"gatv2_sharpe_3bps", "mlp_ir_0bps", "gatv2_break_even_bps"} <= m.keys()


def test_rebalance_dates_without_reference_returns_raise(tmp_path):
    with pytest.raises(ValueError, match="1 rebalance dates have no SPY or BIL"):
        run_backtest({"gatv2": _pred(0)}, _refs(DATES[:-1]), SETTINGS, 5, "1d", tmp_path)


def test_missing_risk_free_ticker_is_reported(tmp_path):
    days = [datetime(2020, 1, 1, 16) + timedelta(days=i) for i in range(10)]
    prices = pl.DataFrame({"timestamp": days, "ticker": ["SPY"] * 10, "open": [1.0] * 10})
    store.write(prices, tmp_path / "benchmarks" / "1d", MANIFEST)
    run_cfg = {"data": {"bar": "1d"}, "horizon_bars": 5}
    with pytest.raises(BenchmarksMissing, match="BIL"):
        reference_returns(run_cfg, SETTINGS, str(tmp_path))


def test_unbuilt_benchmarks_are_reported(tmp_path):
    run_cfg = {"data": {"bar": "1d"}, "horizon_bars": 5}
    with pytest.raises(BenchmarksMissing, match="not built"):
        reference_returns(run_cfg, SETTINGS, str(tmp_path))


def test_backtest_logs_the_benchmarks_manifest_as_opened(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(backtest_run, "mlflow_tracking_uri", lambda: uri)
    days = [DATES[0] - timedelta(days=7) + timedelta(days=i) for i in range(30 * 7 + 21)]
    prices = pl.DataFrame(
        {"timestamp": days * 2, "ticker": ["SPY"] * len(days) + ["BIL"] * len(days)}
    ).with_columns(open=1.0 + pl.int_range(pl.len()) * 1e-4)
    store.write(prices, tmp_path / "benchmarks" / "1d", MANIFEST)
    opened = json.loads((tmp_path / "benchmarks" / "1d" / "manifest.json").read_text())

    mlflow.set_tracking_uri(uri)
    exp = mlflow.create_experiment("t", artifact_location=(tmp_path / "mlruns").as_uri())
    with mlflow.start_run(experiment_id=exp) as r:
        art = tmp_path / "art"
        art.mkdir()
        (art / "config.yaml").write_text(yaml.safe_dump({"horizon_bars": 5, "data": {"bar": "1d"}}))
        _pred(0).write_parquet(art / "predictions_gatv2.parquet")
        mlflow.log_artifacts(str(art))

    backtest_run.backtest(r.info.run_id, {"backtest": SETTINGS, "data_root": str(tmp_path)})

    got = download_artifacts(
        run_id=r.info.run_id, artifact_path="backtest/manifest_benchmarks.json"
    )
    assert json.loads(Path(got).read_text()) == opened
