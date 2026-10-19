"""WalkForwardSplitter: train / validation / test folds on the bar axis, with label purging.

Depends only on the bar axis and config, never on graphs or snapshots. A label at bar i
ends at bar i + 1 + horizon_bars, so each purge gap is horizon_bars + 1 bars.
"""

from dataclasses import dataclass
from datetime import date, datetime, time

import polars as pl


@dataclass(frozen=True)
class Fold:
    """Inclusive timestamp bounds; purge gaps lie between the periods."""

    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime
    holdout: bool


def walk_forward(axis: pl.Series, cfg: dict, horizon_bars: int) -> list[Fold]:
    """Consecutive test periods of `test_bars`, anchored so one starts at `holdout_start`.

    Folds testing on or after `holdout_start` are held out. Test periods are tiled back from
    it as far as a full window allows; bars before the first fold's training window are
    unused in both window modes. The last test period may be shorter.
    """
    window, train, val, test = cfg["window"], cfg["train_bars"], cfg["val_bars"], cfg["test_bars"]
    if window not in ("rolling", "expanding"):
        raise ValueError(f"walk_forward.window must be rolling or expanding, got {window!r}")
    gap = horizon_bars + 1
    lead = train + gap + val + gap  # bars a fold needs before its test period
    ts = axis.sort().to_list()

    holdout_start: date = cfg["holdout_start"]
    cut = datetime.combine(holdout_start, time(), tzinfo=ts[0].tzinfo)
    anchor = next((i for i, t in enumerate(ts) if t >= cut), None)
    if anchor is None:
        raise ValueError(f"holdout_start {holdout_start} is after the last bar {ts[-1]}")
    if anchor - lead < test:
        raise ValueError(
            f"holdout_start {holdout_start} leaves no full search fold before it "
            f"(needs {lead + test} bars, has {anchor})"
        )
    first = anchor - (anchor - lead) // test * test

    folds = []
    for test_start in range(first, len(ts), test):
        val_start = test_start - gap - val
        train_end = val_start - gap - 1
        train_start = first - lead if window == "expanding" else train_end - train + 1
        folds.append(
            Fold(
                train_start=ts[train_start],
                train_end=ts[train_end],
                val_start=ts[val_start],
                val_end=ts[test_start - gap - 1],
                test_start=ts[test_start],
                test_end=ts[min(test_start + test, len(ts)) - 1],
                holdout=test_start >= anchor,
            )
        )
    return folds
