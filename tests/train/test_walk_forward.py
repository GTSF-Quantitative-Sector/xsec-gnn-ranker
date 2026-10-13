from datetime import date, datetime, timedelta

import polars as pl
import pytest

from gnns.train.walk_forward import walk_forward

T0 = datetime(2024, 1, 1, 16)
AXIS = pl.Series([T0 + timedelta(days=i) for i in range(50)])
CFG = {
    "window": "rolling",
    "train_bars": 10,
    "val_bars": 5,
    "test_bars": 8,
    "holdout_start": date(2024, 2, 7),  # bar 37: a boundary of the unshifted layout
}
HOLDOUT = {**CFG, "holdout_start": date(2024, 2, 10)}  # bar 40: shifts the layout


def _pos(t: datetime) -> int:
    return (t - T0).days


def test_first_fold_layout_with_purge_gaps():
    f = walk_forward(AXIS, CFG, horizon_bars=2)[0]  # gap = 2 + 1 = 3 bars
    assert [_pos(t) for t in (f.train_start, f.train_end)] == [0, 9]
    assert [_pos(t) for t in (f.val_start, f.val_end)] == [13, 17]
    assert [_pos(t) for t in (f.test_start, f.test_end)] == [21, 28]


def test_every_label_ends_before_the_next_period_starts():
    h = 2
    for cfg in (CFG, HOLDOUT):
        for f in walk_forward(AXIS, cfg, horizon_bars=h):
            assert _pos(f.train_end) + 1 + h < _pos(f.val_start)
            assert _pos(f.val_end) + 1 + h < _pos(f.test_start)


def test_test_periods_tile_the_rest_of_the_axis():
    folds = walk_forward(AXIS, CFG, horizon_bars=2)
    assert [(_pos(f.test_start), _pos(f.test_end)) for f in folds] == [
        (21, 28),
        (29, 36),
        (37, 44),
        (45, 49),  # last fold is partial
    ]
    assert [f.holdout for f in folds] == [False, False, True, True]


def test_rolling_slides_and_expanding_grows():
    rolling = walk_forward(AXIS, CFG, horizon_bars=2)
    expanding = walk_forward(AXIS, {**CFG, "window": "expanding"}, horizon_bars=2)
    assert [_pos(f.train_start) for f in rolling] == [0, 8, 16, 24]
    assert [_pos(f.train_start) for f in expanding] == [0, 0, 0, 0]
    assert [f.train_end for f in rolling] == [f.train_end for f in expanding]


def test_holdout_anchors_a_test_period_at_its_start():
    folds = walk_forward(AXIS, HOLDOUT, horizon_bars=2)
    # Tiled back from bar 40 in steps of 8, keeping only folds with a full window (lead 21).
    assert [_pos(f.test_start) for f in folds] == [24, 32, 40, 48]
    assert [f.holdout for f in folds] == [False, False, True, True]


def test_bars_before_the_first_window_are_unused_in_both_modes():
    expanding = walk_forward(AXIS, {**HOLDOUT, "window": "expanding"}, horizon_bars=2)
    rolling = walk_forward(AXIS, HOLDOUT, horizon_bars=2)
    assert [_pos(f.train_start) for f in expanding] == [3, 3, 3, 3]
    assert _pos(rolling[0].train_start) == 3


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        ({**CFG, "window": "sliding"}, "rolling or expanding"),
        ({**CFG, "train_bars": 100}, "no full search fold"),
        ({**CFG, "holdout_start": date(2024, 1, 26)}, "no full search fold"),
        ({**CFG, "holdout_start": date(2024, 3, 1)}, "after the last bar"),
    ],
)
def test_bad_config_raises(cfg, message):
    with pytest.raises(ValueError, match=message):
        walk_forward(AXIS, cfg, horizon_bars=2)
