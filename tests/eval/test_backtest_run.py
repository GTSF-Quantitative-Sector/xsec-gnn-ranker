from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from gnns.eval.backtest_run import BenchmarksMissing, reference_returns, run_backtest

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
    m = run_backtest({"gatv2": _pred(0), "mlp": _pred(1)}, _refs(DATES), SETTINGS, 52.0, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "1_equity.png",
        "2_drawdown.png",
        "3_per_year.png",
        "periods_gatv2.parquet",
        "periods_mlp.parquet",
    ]
    assert {"gatv2_sharpe_3bps", "mlp_ir_0bps", "gatv2_break_even_bps"} <= m.keys()


def test_rebalance_dates_without_reference_returns_raise(tmp_path):
    with pytest.raises(ValueError, match="1 rebalance dates have no SPY or BIL"):
        run_backtest({"gatv2": _pred(0)}, _refs(DATES[:-1]), SETTINGS, 52.0, tmp_path)


def test_missing_risk_free_ticker_is_reported(tmp_path):
    dest = tmp_path / "benchmarks" / "1d"
    dest.mkdir(parents=True)
    days = [datetime(2020, 1, 1, 16) + timedelta(days=i) for i in range(10)]
    prices = pl.DataFrame({"timestamp": days, "ticker": ["SPY"] * 10, "open": [1.0] * 10})
    prices.write_parquet(dest / "prices.parquet")
    run_cfg = {"data": {"bar": "1d"}, "horizon_bars": 5}
    with pytest.raises(BenchmarksMissing, match="BIL"):
        reference_returns(run_cfg, SETTINGS, str(tmp_path))
