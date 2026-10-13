"""Build the configured graph's snapshots in memory: load -> features -> build each t."""

from gnns.config import ROOT
from gnns.data import features, store
from gnns.data.snapshot import GraphSnapshot
from gnns.graphs.base import GraphBuilder
from gnns.graphs.sector_etf import SectorETFCorrelationBuilder

BUILDERS: dict[str, type[GraphBuilder]] = {
    "sector_etf_correlation": SectorETFCorrelationBuilder,
}


def build_snapshots(cfg: dict) -> list[GraphSnapshot]:
    """One snapshot every `snapshot_every_bars` bars, wherever any node is eligible."""
    graph = cfg["graph"]
    if graph["builder"] not in BUILDERS:
        raise ValueError(f"unknown builder {graph['builder']!r}; valid: {sorted(BUILDERS)}")

    prices = store.load(ROOT / cfg["data_root"], graph["universe"], cfg["data"]["bar"]).collect()
    feats = features.compute(prices, graph["features"])
    axis = prices["timestamp"].unique().sort()
    builder = BUILDERS[graph["builder"]](
        {"prices": prices, "features": feats}, axis, **graph.get("params", {})
    )
    # A node's own history must cover the builder's price window, or newly eligible nodes
    # get edges from partial windows.
    min_history, need = graph["features"]["min_history_bars"], builder.lookbacks().get("prices")
    if need is not None and min_history < need - 1:
        raise ValueError(
            f"graph features.min_history_bars={min_history} is shorter than {graph['builder']}'s "
            f"price window ({need - 1} returns); newly eligible nodes would get partial windows"
        )

    eligible = set(feats["timestamp"])
    return [
        builder.build(t) for t in axis.gather_every(cfg["snapshot_every_bars"]) if t in eligible
    ]
