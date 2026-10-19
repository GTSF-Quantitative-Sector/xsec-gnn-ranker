"""Dummy builder: fully connected sector ETFs, weighted by rolling return correlation."""

from datetime import datetime

import numpy as np
import polars as pl

from gnns.data.snapshot import GraphSnapshot, add_reverse_edges
from gnns.graphs.base import GraphBuilder
from gnns.graphs.stats import pairwise_corr


class SectorETFCorrelationBuilder(GraphBuilder):
    def __init__(
        self, inputs: dict[str, pl.DataFrame], axis: pl.Series, *, window_bars: int
    ) -> None:
        self.window_bars = window_bars
        super().__init__(inputs, axis)

    def lookbacks(self) -> dict[str, int | None]:
        # window_bars returns need window_bars + 1 closes.
        return {"prices": self.window_bars + 1, "features": 1}

    def build_at(self, t: datetime, inputs: dict[str, pl.DataFrame]) -> GraphSnapshot:
        feats = inputs["features"].sort("ticker")
        nodes = feats["ticker"].to_list()
        feature_names = [c for c in feats.columns if c not in ("timestamp", "ticker")]

        wide = inputs["prices"].pivot(on="ticker", index="timestamp", values="close")
        closes = (
            wide.sort("timestamp")
            .select(
                pl.col(n) if n in wide.columns else pl.lit(None, pl.Float64).alias(n) for n in nodes
            )
            .to_numpy()
            .astype(np.float64)
        )
        rets = closes[1:] / closes[:-1] - 1

        # Each unordered pair once; skip pairs with < 3 shared returns or no variance.
        corr, n_obs = pairwise_corr(rets)
        i, j = np.triu_indices(len(nodes), k=1)
        keep = (n_obs[i, j] >= 3) & np.isfinite(corr[i, j])
        i, j = i[keep], j[keep]
        c = corr[i, j].astype(np.float32)
        edge_index, edge_weight, edge_type, edge_attr = add_reverse_edges(
            np.stack([i, j]).astype(np.int64),
            np.abs(c),
            np.zeros(len(c), dtype=np.int64),
            c.reshape(-1, 1),
        )
        n = len(nodes)
        return GraphSnapshot(
            timestamp=t,
            node_ids=np.array(nodes, dtype=str),
            node_type=np.zeros(n, dtype=np.int64),
            node_type_names=("equity",),
            target_mask=np.ones(n, dtype=bool),
            x=feats.select(feature_names).to_numpy().astype(np.float32),
            feature_names=np.array(feature_names, dtype=str),
            feature_owner=np.zeros(len(feature_names), dtype=np.int64),
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_type=edge_type,
            edge_type_names=("correlated",),
            edge_attr=edge_attr,
            edge_attr_names=np.array(["corr"]),
        )
