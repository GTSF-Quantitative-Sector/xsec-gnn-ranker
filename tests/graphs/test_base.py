from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from gnns.data.snapshot import GraphSnapshot
from gnns.graphs.base import GraphBuilder

T0 = datetime(2024, 1, 1, 16, tzinfo=ZoneInfo("America/New_York"))
AXIS = pl.Series([T0 + timedelta(days=i) for i in range(6)])


def _empty(t: datetime) -> GraphSnapshot:
    return GraphSnapshot(
        timestamp=t,
        node_ids=np.array(["A"]),
        node_type=np.zeros(1, dtype=np.int64),
        node_type_names=("equity",),
        target_mask=np.ones(1, dtype=bool),
        x=np.zeros((1, 0), dtype=np.float32),
        feature_names=np.array([], dtype=str),
        feature_owner=np.zeros(0, dtype=np.int64),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_weight=np.zeros(0, dtype=np.float32),
        edge_type=np.zeros(0, dtype=np.int64),
        edge_type_names=("none",),
        edge_attr=np.zeros((0, 0), dtype=np.float32),
        edge_attr_names=np.array([], dtype=str),
    )


class Probe(GraphBuilder):
    """Records what it was given."""

    def lookbacks(self) -> dict[str, int | None]:
        return {"bars": 2, "events": None}

    def build_at(self, t, inputs):
        self.seen = {name: df["timestamp"].to_list() for name, df in inputs.items()}
        return _empty(t)


def _inputs() -> dict[str, pl.DataFrame]:
    bars = pl.DataFrame({"timestamp": AXIS, "v": range(6)})
    # Irregular rows: one exactly on a bar, the rest between bars.
    events = pl.DataFrame(
        {"timestamp": [AXIS[1], AXIS[1] + timedelta(hours=3), AXIS[4] + timedelta(hours=1)]}
    )
    return {"bars": bars.reverse(), "events": events}  # unsorted on purpose


def test_slices_each_input_to_its_window_ending_at_t():
    probe = Probe(_inputs(), AXIS)
    probe.build(AXIS[3])
    assert probe.seen["bars"] == [AXIS[2], AXIS[3]]  # the 2 bars ending at t
    assert probe.seen["events"] == [AXIS[1], AXIS[1] + timedelta(hours=3)]  # all <= t


def test_irregular_rows_fall_into_the_next_bar():
    probe = Probe(_inputs(), AXIS)
    probe.build(AXIS[4])
    assert AXIS[4] + timedelta(hours=1) not in probe.seen["events"]
    probe.build(AXIS[5])
    assert AXIS[4] + timedelta(hours=1) in probe.seen["events"]


def test_early_t_gets_all_available_history():
    probe = Probe(_inputs(), AXIS)
    probe.build(AXIS[0])
    assert probe.seen["bars"] == [AXIS[0]]


def test_missing_input_and_off_axis_t_raise():
    with pytest.raises(ValueError, match="needs inputs"):
        Probe({"bars": _inputs()["bars"]}, AXIS)
    with pytest.raises(ValueError, match="not on the bar axis"):
        Probe(_inputs(), AXIS).build(AXIS[0] + timedelta(hours=1))


def test_wrong_snapshot_timestamp_raises():
    class Liar(Probe):
        def build_at(self, t, inputs):
            return _empty(AXIS[0])

    with pytest.raises(ValueError, match="stamped"):
        Liar(_inputs(), AXIS).build(AXIS[2])
