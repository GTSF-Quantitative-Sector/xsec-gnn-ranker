"""GraphBuilder base class: slices every input to what is known at t, then builds."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import polars as pl

from gnns.data.snapshot import GraphSnapshot


class GraphBuilder(ABC):
    """Subclasses declare `lookbacks()` and implement `build_at`; callers use `build`.

    `inputs` are point-in-time tables, each with a known-at `timestamp` column
    (docs/CONTRACTS.md, "Point-in-time inputs"). `axis` is the universe's bar axis.
    A builder with extra datasets or config requirements overrides `load_sources` and
    `check_config`; `build_snapshots` calls both, so shared code never names a builder.
    """

    @classmethod  # noqa: B027 (optional hook; the default accepts every config)
    def check_config(cls, cfg: dict) -> None:
        """Raise if the resolved config can't work with this construction.

        Runs before any data is loaded, so it sees config values only (e.g. `normalize`,
        `horizon_bars`, required `sources:` keys); data checks belong in `load_sources`.
        """

    @classmethod
    def load_sources(cls, root: Path, sources: dict) -> dict[str, tuple[pl.DataFrame, dict]]:
        """Extra point-in-time inputs beyond prices and features, from the graph file's
        `sources:`, each with the manifest it was opened with (`gnns.data.store.open`).

        `root` is the data root. Loaders live in `gnns.data`; this method calls them.
        """
        return {}

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
