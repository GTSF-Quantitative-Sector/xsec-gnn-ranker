from datetime import datetime, timedelta

import numpy as np
import polars as pl

from gnns.eval.report import make_charts

CFG = {
    "eval": {"quantiles": 3, "top_k": 2, "benchmark": "SPY", "rolling_dates": 3},
    "horizon_bars": 5,
}


def _pred(seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = datetime(2020, 1, 3, 16)
    rows = [
        (
            t0 + timedelta(weeks=w),
            ticker,
            float(rng.normal()),
            float(rng.normal(0, 0.02)),
            w // 10 + 1,
        )
        for w in range(30)
        for ticker in "ABCDEF"
    ]
    return pl.DataFrame(
        rows, schema=["timestamp", "ticker", "score", "fwd_return", "fold"], orient="row"
    )


def test_make_charts_writes_every_png(tmp_path):
    preds = {"gatv2": _pred(0), "mlp": _pred(1)}
    hist = [[{"epoch": e, "train_loss": 1.0 / e, "val_ic": 0.01 * e} for e in range(1, 4)]] * 3
    bench = _pred(2).filter(pl.col("ticker") == "A").select("timestamp", ret="fwd_return")
    make_charts(preds, {"gatv2": hist, "mlp": hist}, CFG, bench, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "10_training_curves.png",
        "1_ic_over_time.png",
        "3_cumulative_ic.png",
        "4_ic_by_fold.png",
        "5_quantile_cascade.png",
        "6_quantile_curves.png",
        "7_equity_curves.png",
    ]
