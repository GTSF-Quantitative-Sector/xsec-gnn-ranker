"""fetch_macro against a fake Massive client: joins, adjustment, checks, look-ahead."""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

import gnns.data.massive as massive
from gnns.data import store
from gnns.data.macro import FACTORS, UNITS, fetch_macro, load_macro

NY = ZoneInfo("America/New_York")
DAYS = [date(2019, 1, 1) + timedelta(days=i) for i in range(60)]  # synthetic calendar
START, END = DAYS[0], DAYS[-1]


def _closes(seed: int, n: int, level: float) -> np.ndarray:
    return level * np.cumprod(1 + np.random.default_rng(seed).normal(0, 0.01, n))


def market() -> dict[str, dict[date, tuple[float, float]]]:
    """ticker -> date -> (adjusted close, unadjusted close).

    OLD runs to day 20 then is replaced by NEW_B (bars from day 10, so the two overlap);
    NEW_B is renamed OLD after day 40: OLD's bars resume on day 41 with no day-40 bar,
    adjusted for a later 1:64 reverse split.
    """
    m: dict[str, dict[date, tuple[float, float]]] = {}
    for k, t in enumerate(["R", "C", "O", "G"]):
        c = _closes(k, 60, 50.0 + 10 * k)
        m[t] = {d: (float(v), float(v)) for d, v in zip(DAYS, c, strict=True)}
    old, b = _closes(10, 21, 2000.0), _closes(11, 31, 30.0)
    m["NEW_B"] = {DAYS[10 + i]: (float(v), float(v)) for i, v in enumerate(b)}
    tail = b[-1] * _closes(12, 19, 1.0) / _closes(12, 19, 1.0)[0] * 1.002
    m["OLD"] = {DAYS[i]: (float(v), float(v)) for i, v in enumerate(old)}
    m["OLD"] |= {DAYS[41 + i]: (float(v * 64), float(v)) for i, v in enumerate(tail)}
    return m


SPEC = {
    "series": {
        "rates": {"ticker": "R"},
        "credit": {"ticker": "C"},
        "vix": {
            "stitch": [
                {"ticker": "OLD", "until": DAYS[20]},
                {"ticker": "NEW_B", "until": DAYS[40]},
                {"ticker": "OLD"},
            ]
        },
        "oil": {"ticker": "O"},
        "gold": {"ticker": "G"},
    }
}
DIVIDEND = ("R", DAYS[30], 0.99)  # ex-date day 30: earlier R bars are scaled by 0.99


@pytest.fixture
def fake_massive(monkeypatch):
    data, calls = market(), []

    def daily_bars(ticker, start, end, adjusted=True):
        calls.append(ticker)
        rows = [
            (datetime.combine(d, time(16), NY), ticker, *(4 * [v[0] if adjusted else v[1]]), 1.0)
            for d, v in sorted(data[ticker].items())
            if start <= d <= end
        ]
        return pl.DataFrame(rows, schema=store.PRICE_COLS, orient="row")

    def dividends(ticker):
        rows = [(t, x, f) for t, x, f in [DIVIDEND] if t == ticker]
        return pl.DataFrame(
            rows,
            schema={"ticker": pl.String, "ex_date": pl.Date, "factor": pl.Float64},
            orient="row",
        )

    monkeypatch.setattr(massive, "daily_bars", daily_bars)
    monkeypatch.setattr(massive, "dividends", dividends)
    return data, calls


def _series(events: pl.DataFrame, factor: str) -> pl.DataFrame:
    return events.filter(pl.col("factor") == factor).select(
        pl.col("timestamp").dt.date().alias("date"), "value"
    )


def test_fetch_joins_segments_on_returns_and_adjusts_dividends(fake_massive):
    data, _ = fake_massive
    events, extras = fetch_macro(SPEC, START, END)
    assert set(events["factor"]) == set(FACTORS)
    assert events["timestamp"].dt.time().unique().to_list() == [time(16)]
    assert str(events.schema["timestamp"].time_zone) == "America/New_York"  # type: ignore[union-attr]

    vix = _series(events, "vix")
    assert vix["date"].to_list() == DAYS  # one bar per day, no gap at either switch
    r = (vix["value"] / vix["value"].shift(1) - 1).to_numpy()

    def own(t, d):  # a ticker's own adjusted return into day d
        prev = max(x for x in data[t] if x < d)
        return data[t][d][0] / data[t][prev][0] - 1

    assert vix["value"][0] == pytest.approx(data["OLD"][DAYS[0]][0])  # first segment's prices
    for i in range(1, 60):
        if i <= 20:
            want = own("OLD", DAYS[i])
        elif i <= 40:
            want = own("NEW_B", DAYS[i])  # day 21: NEW_B's own return from its day-20 bar
        elif i == 41:  # renamed note: unadjusted OLD on day 41 vs NEW_B on day 40
            want = data["OLD"][DAYS[41]][1] / data["NEW_B"][DAYS[40]][1] - 1
        else:
            want = own("OLD", DAYS[i])
        assert r[i] == pytest.approx(want), DAYS[i]

    rates = _series(events, "rates")
    raw = np.array([data["R"][d][0] for d in DAYS])
    want = np.where(np.array(DAYS) < DIVIDEND[1], raw * DIVIDEND[2], raw)
    assert rates["value"].to_numpy() == pytest.approx(want)

    assert extras["units"] == UNITS and extras["known_at_verified"] is True
    assert extras["series"]["vix"]["stitch"][0]["until"] == str(DAYS[20])  # JSON-safe
    assert extras["coverage"]["vix"] == [str(START), str(END)]
    assert "OLD/NEW_B/OLD" in extras["source"]


def test_written_dataset_loads(fake_massive, tmp_path):
    events, extras = fetch_macro(SPEC, START, END)
    manifest = extras | {"built_at": "2026-01-01T00:00:00+00:00", "git_sha": "test"}
    store.write(events, tmp_path / "macro", manifest, file="macro.parquet")
    loaded, meta = load_macro(tmp_path, {"path": "macro"})
    assert loaded.equals(events)
    assert meta["series"] == extras["series"]


def test_broken_join_rejected(fake_massive):
    data, _ = fake_massive
    for d in DAYS[21:41]:  # NEW_B jumps 50% on the first day it is used
        adj, raw = data["NEW_B"][d]
        data["NEW_B"][d] = (adj * 1.5, raw * 1.5)
    with pytest.raises(ValueError, match="join is broken"):
        fetch_macro(SPEC, START, END)


def test_segment_ending_before_its_until_rejected(fake_massive):
    data, _ = fake_massive
    del data["OLD"][DAYS[20]]
    with pytest.raises(ValueError, match="ends before"):
        fetch_macro(SPEC, START, END)


def test_spec_must_define_every_factor(fake_massive):
    spec = {"series": {k: v for k, v in SPEC["series"].items() if k != "gold"}}
    with pytest.raises(ValueError, match="exactly"):
        fetch_macro(spec, START, END)


def test_deleting_later_data_changes_no_earlier_event(fake_massive):
    full, _ = fetch_macro(SPEC, START, END)
    for t in (DAYS[15], DAYS[35], DAYS[50]):  # inside each vix segment
        short, _ = fetch_macro(SPEC, START, t)
        cut = pl.col("timestamp").dt.date() <= t
        assert short.equals(full.filter(cut)), t
