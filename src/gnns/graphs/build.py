"""Build the configured graph's snapshots in memory: load -> features -> build each t."""

from gnns.config import ROOT
from gnns.data import features, store
from gnns.data.macro import load_macro
from gnns.data.snapshot import GraphSnapshot
from gnns.graphs.base import GraphBuilder
from gnns.graphs.macro import MacroGraphBuilder
from gnns.graphs.sector_etf import SectorETFCorrelationBuilder

BUILDERS: dict[str, type[GraphBuilder]] = {
    "sector_etf_correlation": SectorETFCorrelationBuilder,
    "macro_sensitivity": MacroGraphBuilder,
}


def build_snapshots(cfg: dict) -> list[GraphSnapshot]:
    """One snapshot every `horizon_bars` bars, wherever any node is eligible."""
    graph = cfg["graph"]
    if graph["builder"] not in BUILDERS:
        raise ValueError(f"unknown builder {graph['builder']!r}; valid: {sorted(BUILDERS)}")

    prices = store.load(ROOT / cfg["data_root"], graph["universe"], cfg["data"]["bar"]).collect()
    feats = features.compute(prices, graph["features"])
    axis = prices["timestamp"].unique().sort()
    inputs = {"prices": prices, "features": feats}
    if graph["builder"] == "macro_sensitivity":
        if cfg["normalize"] != "none":
            raise ValueError(
                "macro_sensitivity requires normalize: none; builder applies causal macro scaling"
            )
        inputs["macro"] = load_macro(ROOT / cfg["data_root"], graph["sources"]["macro"])
    builder = BUILDERS[graph["builder"]](inputs, axis, **graph.get("params", {}))
    # A node's own history must cover the builder's price window, or newly eligible nodes
    # get edges from partial windows.
    min_history, need = graph["features"]["min_history_bars"], builder.lookbacks().get("prices")
    if need is not None and min_history < need - 1:
        raise ValueError(
            f"graph features.min_history_bars={min_history} is shorter than {graph['builder']}'s "
            f"price window ({need - 1} returns); newly eligible nodes would get partial windows"
        )

    eligible = set(feats["timestamp"])
    snapshots = [builder.build(t) for t in axis.gather_every(cfg["horizon_bars"]) if t in eligible]
    snapshots = [s for s in snapshots if s.target_mask.any()]
    if not snapshots:
        raise ValueError("No eligible graph snapshots; check macro coverage, history and staleness")
    return snapshots
