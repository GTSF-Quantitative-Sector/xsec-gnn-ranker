"""Build the configured graph's snapshots in memory: load -> features -> build each t."""

import logging
from typing import NamedTuple

import polars as pl

from gnns.config import ROOT
from gnns.data import features, store
from gnns.data.snapshot import GraphSnapshot
from gnns.graphs.base import GraphBuilder
from gnns.graphs.macro import MacroGraphBuilder
from gnns.graphs.oil_supply_chain import OilSupplyChainBuilder
from gnns.graphs.sector_etf import SectorETFCorrelationBuilder

log = logging.getLogger(__name__)
BUILDERS: dict[str, type[GraphBuilder]] = {
    "sector_etf_correlation": SectorETFCorrelationBuilder,
    "macro_sensitivity": MacroGraphBuilder,
    "oil_supply_chain": OilSupplyChainBuilder,
}


class Snapshots(NamedTuple):
    snapshots: list[GraphSnapshot]
    manifests: dict[str, dict]  # each dataset opened (`prices`, sources), as read then
    prices: pl.DataFrame  # the universe's prices the snapshots were built from


def build_snapshots(cfg: dict) -> Snapshots:
    """One snapshot every `horizon_bars` bars where any node is eligible and the builder
    returns at least one target node, plus the manifest of every dataset opened (`prices`
    and each of the builder's sources), as read at open time, and the prices (for labels).

    Dates whose snapshot has no target node are dropped and logged; raises if none remain.
    """
    graph = cfg["graph"]
    if graph["builder"] not in BUILDERS:
        raise ValueError(f"unknown builder {graph['builder']!r}; valid: {sorted(BUILDERS)}")
    cls = BUILDERS[graph["builder"]]
    cls.check_config(cfg)  # before any I/O

    root = ROOT / cfg["data_root"]
    lf, price_manifest = store.open(root / graph["universe"] / cfg["data"]["bar"])
    prices = lf.collect()
    feats = features.compute(prices, graph["features"])
    axis = prices["timestamp"].unique().sort()
    sources = cls.load_sources(root, graph.get("sources", {}))
    if clash := sorted(sources.keys() & {"prices", "features"}):
        raise ValueError(f"{graph['builder']}: source names {clash} are reserved")
    inputs = {"prices": prices, "features": feats} | {k: df for k, (df, _) in sources.items()}
    manifests = {"prices": price_manifest} | {k: m for k, (_, m) in sources.items()}
    builder = cls(inputs, axis, **graph.get("params", {}))
    # A node's own history must cover the builder's price window, or newly eligible nodes
    # get edges from partial windows.
    min_history, need = graph["features"]["min_history_bars"], builder.lookbacks().get("prices")
    if need is not None and min_history < need - 1:
        raise ValueError(
            f"graph features.min_history_bars={min_history} is shorter than {graph['builder']}'s "
            f"price window ({need - 1} returns); newly eligible nodes would get partial windows"
        )

    eligible = set(feats["timestamp"])
    built = [builder.build(t) for t in axis.gather_every(cfg["horizon_bars"]) if t in eligible]
    snapshots = [s for s in built if s.target_mask.any()]
    if dropped := [s.timestamp for s in built if not s.target_mask.any()]:
        log.info(
            "%s: dropped %d of %d dates with no target node (%s .. %s)",
            graph["builder"],
            len(dropped),
            len(built),
            dropped[0],
            dropped[-1],
        )
    if not snapshots:
        raise ValueError(
            f"{graph['builder']}: no snapshot has a target node; check its input coverage "
            "and history"
        )
    return Snapshots(snapshots, manifests, prices)
