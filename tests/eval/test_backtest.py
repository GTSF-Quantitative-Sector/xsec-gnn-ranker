from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from gnns.eval import backtest, metrics

T0 = datetime(2020, 1, 3, 16)


def _pred(rows: list[tuple]) -> pl.DataFrame:
    """rows: (week, ticker, score, fwd_return)."""
    return pl.DataFrame(
        [(T0 + timedelta(weeks=w), k, s, r) for w, k, s, r in rows],
        schema=["timestamp", "ticker", "score", "fwd_return"],
        orient="row",
    )


def _periods(gross, traded, bench, rf, start=T0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(weeks=i) for i in range(len(gross))],
            "gross": gross,
            "traded": traded,
            "benchmark": bench,
            "risk_free": rf,
        }
    )


# gross, traded, benchmark: equity 1.02, 1.0098, 1.0401, 1.0193; mean active 0.0025
GROSS, TRADED, BENCH = [0.02, -0.01, 0.03, -0.02], [1.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.01, 0.0]


def test_equal_weight_top_k_and_fewer_names_than_k():
    p = _pred([(0, "A", 3.0, 0.0), (0, "B", 2.0, 0.0), (0, "C", 1.0, 0.0), (1, "A", 1.0, 0.0)])
    w = backtest.target_weights(p, 2, "equal").sort("timestamp", "ticker")
    assert w["ticker"].to_list() == ["A", "B", "A"]
    assert w["weight"].to_list() == [0.5, 0.5, 1.0]


def test_gross_return_and_turnover_after_drift():
    # Week 0 holds A (+10%) and B (0%): gross 5%; A drifts to 0.55/1.05, B to 0.5/1.05.
    # Week 1 targets A and C at 0.5: traded = |0.5 - 0.55/1.05| + 0.5/1.05 + 0.5 = 1.0.
    p = _pred(
        [
            (0, "A", 3.0, 0.10),
            (0, "B", 2.0, 0.0),
            (0, "C", 1.0, -0.2),
            (1, "A", 3.0, 0.02),
            (1, "B", 1.0, 0.5),
            (1, "C", 2.0, 0.04),
        ]
    )
    s = backtest.simulate(p, 2, "equal")
    assert s["gross"].to_list() == pytest.approx([0.05, 0.03])
    assert s["traded"].to_list() == pytest.approx([1.0, 1.0])  # week 0 buys from cash


def test_unchanged_holdings_trade_only_the_drift():
    # A +10%, B -10% -> drifted 0.55 / 0.45 -> back to 0.5 / 0.5 trades 0.1.
    p = _pred([(0, "A", 2.0, 0.1), (0, "B", 1.0, -0.1), (1, "A", 2.0, 0.0), (1, "B", 1.0, 0.0)])
    assert backtest.simulate(p, 2, "equal")["traded"].to_list() == pytest.approx([1.0, 0.1])


def test_gross_matches_evaluator_top_k():
    rng = np.random.default_rng(0)
    p = _pred(
        [
            (w, k, float(rng.normal()), float(rng.normal(0, 0.02)))
            for w in range(20)
            for k in "ABCDEFG"
        ]
    )
    gross = backtest.simulate(p, 3, "equal")["gross"].to_list()
    assert gross == pytest.approx(metrics.top_k_returns(p, 3)["ret"].to_list())


def test_held_name_without_return_raises_unheld_is_ignored():
    p = _pred([(0, "A", 2.0, None), (0, "B", 1.0, 0.0), (0, "C", 0.0, None)])
    with pytest.raises(ValueError, match=r"(?s)1 held positions.*A"):
        backtest.simulate(p, 2, "equal")
    without_a = p.filter(pl.col("ticker") != "A")
    assert backtest.simulate(without_a, 1, "equal")["gross"].to_list() == [0.0]


def test_summary_known_answers():
    m = backtest.summarize(_periods(GROSS, TRADED, BENCH, [0.001] * 4), [0, 10], per_year=52)
    assert m["turnover"] == pytest.approx(0.25)  # mean traded 0.5, one-way
    assert m["turnover_annual"] == pytest.approx(13.0)
    assert m["break_even_bps"] == pytest.approx(50.0)  # mean active 0.0025 / mean traded 0.5
    assert m["ann_return_0bps"] == pytest.approx((1.02 * 0.99 * 1.03 * 0.98) ** 13 - 1)
    assert m["max_drawdown_0bps"] == pytest.approx(-0.02)  # 1.0401 -> 1.0193
    assert m["max_dd_days_0bps"] == pytest.approx(252 / 52)  # one period underwater at a time
    assert m["excess_return_0bps"] == pytest.approx(0.0025 * 52)
    assert m["win_rate_0bps"] == 0.5
    assert m["win_rate_vs_benchmark_0bps"] == 0.5
    assert m["cost_drag_10bps"] == pytest.approx(0.001 * 0.5 * 52)
    ex = np.array([0.019, -0.011, 0.029, -0.021])
    assert m["sharpe_0bps"] == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(52))


def test_break_even_cost_zeroes_the_excess_return():
    p = _periods(GROSS, TRADED, BENCH, [0.0] * 4)
    be = backtest.summarize(p, [], per_year=52)["break_even_bps"]
    m = backtest.summarize(p, [be], per_year=52)
    assert m[f"excess_return_{be:g}bps"] == pytest.approx(0, abs=1e-12)


def test_yearly_compounds_within_calendar_years():
    p = _periods(GROSS, [0.0] * 4, [0.0] * 4, [0.0] * 4, start=datetime(2020, 12, 18, 16))
    y = backtest.yearly(p, 0, per_year=52)
    assert y["year"].to_list() == [2020, 2021]
    assert y["net_return"].to_list() == pytest.approx([1.02 * 0.99 - 1, 1.03 * 0.98 - 1])
