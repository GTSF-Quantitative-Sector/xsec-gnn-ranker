"""GraphBuilder base class: slices every input to what is known at t, then builds."""

from abc import ABC, abstractmethod
from datetime import datetime

import polars as pl

from gnns.data.snapshot import GraphSnapshot


class GraphBuilder(ABC):
    """Subclasses declare `lookbacks()` and implement `build_at`; callers use `build`.

    `inputs` are point-in-time tables, each with a known-at `timestamp` column
    (docs/CONTRACTS.md, "Point-in-time inputs"). `axis` is the universe's bar axis.
    """

    def __init__(self, inputs: dict[str, pl.DataFrame], axis: pl.Series) -> None:
        if missing := sorted(self.lookbacks().keys() - inputs.keys()):
            raise ValueError(f"{type(self).__name__} needs inputs {missing}")
        self._inputs = {name: inputs[name].sort("timestamp") for name in self.lookbacks()}
        self._axis = axis.sort()

    @abstractmethod
    def lookbacks(self) -> dict[str, int | None]:
        """Bars of history each input needs, ending at t (`None` = everything up to t)."""

    @abstractmethod
    def build_at(self, t: datetime, inputs: dict[str, pl.DataFrame]) -> GraphSnapshot:
        """Build the snapshot at `t` from inputs already sliced to rows known by `t`."""

    def build(self, t: datetime) -> GraphSnapshot:
        i = self._axis.search_sorted(t)
        if i >= len(self._axis) or self._axis[i] != t:
            raise ValueError(f"{t} is not on the bar axis")
        sliced = {}
        for name, lookback in self.lookbacks().items():
            ts = self._inputs[name]["timestamp"]
            lo = (
                0
                if lookback is None or i < lookback
                else ts.search_sorted(self._axis[i - lookback], side="right")
            )
            hi = ts.search_sorted(t, side="right")
            sliced[name] = self._inputs[name].slice(lo, hi - lo)
        snapshot = self.build_at(t, sliced)
        if snapshot.timestamp != t:
            raise ValueError(f"build_at({t}) returned a snapshot stamped {snapshot.timestamp}")
        return snapshot
