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
    hist = [
        [
            {"epoch": e, "train_loss": 1.0 / e, "val_ic": 0.01 * e, "no_info_loss": 0.5}
            for e in (1, 2)
        ]
    ] * 3
    bench = _pred(2).filter(pl.col("ticker") == "A").select("timestamp", ret="fwd_return")
    imp = [{"mom": 0.05, "vol": -0.01, "edges": 0.02}] * 3
    corr = (np.array([[1.0, 0.4], [0.4, 1.0]]), ["mom", "vol"])
    make_charts(
        preds, {"gatv2": hist, "mlp": hist}, CFG, bench, tmp_path, {"gatv2": imp, "mlp": imp}, corr
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "10_training_curves.png",
        "11_feature_correlation.png",
        "1_ic_over_time.png",
        "3_cumulative_ic.png",
        "4_ic_by_fold.png",
        "5_quantile_cascade.png",
        "6_quantile_curves.png",
        "7_equity_curves.png",
        "8_score_spread.png",
        "9_feature_importance.png",
    ]
