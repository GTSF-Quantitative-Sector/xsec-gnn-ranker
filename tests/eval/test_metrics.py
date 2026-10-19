import math
from datetime import datetime

import polars as pl
import pytest

from gnns.eval import metrics

T1, T2 = datetime(2024, 1, 5, 16), datetime(2024, 1, 12, 16)


def _pred() -> pl.DataFrame:
    # T1: scores rank perfectly (IC 1); T2: perfectly reversed (IC -1).
    return pl.DataFrame(
        {
            "timestamp": [T1] * 4 + [T2] * 4,
            "ticker": list("ABCD") * 2,
            "score": [4.0, 3.0, 2.0, 1.0] * 2,
            "fwd_return": [0.04, 0.03, 0.02, 0.01, -0.04, -0.03, -0.02, -0.01],
        }
    )


def test_daily_ic_and_summary():
    ic = metrics.daily_ic(_pred())
    assert ic["ic"].to_list() == pytest.approx([1.0, -1.0])
    s = metrics.ic_summary(ic)
    assert s["mean_ic"] == pytest.approx(0.0) and s["hit_rate"] == 0.5 and s["n_dates"] == 2


def test_icir_annual_and_its_relation_to_ic_t():
    ic = pl.DataFrame({"ic": [0.1, 0.3, 0.2, 0.0]})  # mean 0.15, std 0.129
    icir = 0.15 / 0.12909944
    s = metrics.ic_summary(ic, periods_per_year=52.0)
    assert s["icir_annual"] == pytest.approx(icir * math.sqrt(52))
    assert s["ic_t"] == pytest.approx(icir * math.sqrt(4))
    years = 4 / 52
    assert s["ic_t"] == pytest.approx(s["icir_annual"] * math.sqrt(years))


@pytest.mark.parametrize(("bar", "expected"), [("1d", 252), ("5d", 50.4), ("65m", 6 * 252)])
def test_bars_per_year(bar, expected):
    assert metrics.bars_per_year(bar) == pytest.approx(expected)


def test_quantiles_top_k_and_universe():
    p = _pred()
    q = metrics.quantile_returns(p, 2).filter(pl.col("timestamp") == T1)
    assert q["ret"].to_list() == pytest.approx([0.035, 0.015])  # Q1 = top scores
    assert metrics.top_k_returns(p, 1)["ret"].to_list() == pytest.approx([0.04, -0.04])
    assert metrics.universe_returns(p)["ret"].to_list() == pytest.approx([0.025, -0.025])
