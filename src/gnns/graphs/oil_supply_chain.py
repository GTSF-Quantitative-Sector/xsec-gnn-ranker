"""Directed oil exposure graph with a context-only Brent futures proxy."""

from datetime import datetime

import numpy as np
import polars as pl

from gnns.data.snapshot import GraphSnapshot
from gnns.graphs.base import GraphBuilder

EDGE_TYPES = ("producer_revenue", "crude_feedstock", "jet_fuel", "road_fuel")


class OilSupplyChainBuilder(GraphBuilder):
    def __init__(
        self,
        inputs: dict[str, pl.DataFrame],
        axis: pl.Series,
        *,
        oil_ticker: str,
        oil_return_bars: list[int],
        edges: dict[str, list[list[str]]],
    ) -> None:
        if (
            not oil_return_bars
            or any(type(b) is not int or b <= 0 for b in oil_return_bars)
            or len(set(oil_return_bars)) != len(oil_return_bars)
        ):
            raise ValueError("oil_return_bars must contain distinct positive integers")
        self.oil_ticker = oil_ticker
        self.oil_return_bars = oil_return_bars
        self.edges: list[tuple[str, str, int]] = []
        for kind, pairs in edges.items():
            if kind not in EDGE_TYPES:
                raise ValueError(f"unknown oil edge type {kind!r}")
            for src, dst in pairs:
                direct = kind in ("producer_revenue", "crude_feedstock")
                if src == dst or dst == oil_ticker or (src == oil_ticker) != direct:
                    raise ValueError(f"invalid oil edge {src} -> {dst} ({kind})")
                self.edges.append((src, dst, EDGE_TYPES.index(kind)))
        if len({(src, dst) for src, dst, _ in self.edges}) != len(self.edges):
            raise ValueError("duplicate oil edge endpoints")
        # Sweep validation constructs builders on an empty axis.
        if len(axis):
            tickers = set(inputs["prices"]["ticker"])
            required = {oil_ticker} | {
                ticker for src, dst, _ in self.edges for ticker in (src, dst)
            }
            if missing := sorted(required - tickers):
                raise ValueError(f"oil supply chain is missing prices for {missing}")
        super().__init__(inputs, axis)

    def lookbacks(self) -> dict[str, int | None]:
        return {"prices": max(self.oil_return_bars) + 1, "features": 1}

    def build_at(self, t: datetime, inputs: dict[str, pl.DataFrame]) -> GraphSnapshot:
        prices = inputs["prices"]
        equity = inputs["features"].filter(pl.col("ticker") != self.oil_ticker)
        equity = equity.join(prices.filter(pl.col("timestamp") == t), on="ticker", how="semi").sort(
            "ticker"
        )
        equity_features = [c for c in equity.columns if c not in ("timestamp", "ticker")]
        i = self._axis.search_sorted(t)
        oil = dict(
            prices.filter(pl.col("ticker") == self.oil_ticker)
            .select("timestamp", "close")
            .iter_rows()
        )
        closes = np.array(
            [oil.get(t, np.nan)]
            + [
                oil.get(self._axis[i - b], np.nan) if i >= b else np.nan
                for b in self.oil_return_bars
            ]
        )
        ready = (np.isfinite(closes) & (closes > 0)).all() and not equity.is_empty()
        nodes = equity["ticker"].to_list() + [self.oil_ticker] if ready else []
        n = len(nodes)
        k = len(self.oil_return_bars)
        equity_width = len(equity_features) + k
        x = np.zeros((n, equity_width + k), dtype=np.float32)
        if ready:
            returns = closes[0] / closes[1:] - 1
            x[:-1, : len(equity_features)] = equity.select(equity_features).to_numpy()
            x[:-1, len(equity_features) : equity_width] = returns
            x[-1, equity_width:] = returns

        pos = {ticker: j for j, ticker in enumerate(nodes)}
        edges = [(pos[s], pos[d], kind) for s, d, kind in self.edges if s in pos and d in pos]
        edge_index = np.array([(src, dst) for src, dst, _ in edges], dtype=np.int64).reshape(-1, 2)
        edge_type = np.array([kind for _, _, kind in edges], dtype=np.int64)
        target = np.arange(n) < n - 1
        return GraphSnapshot(
            timestamp=t,
            node_ids=np.array(nodes, dtype=str),
            node_type=(~target).astype(np.int64),
            node_type_names=("equity", "oil"),
            target_mask=target,
            x=x,
            feature_names=np.array(
                equity_features
                + [f"oil_return_{b}" for b in self.oil_return_bars]
                + [f"oil_context_return_{b}" for b in self.oil_return_bars]
            ),
            feature_owner=np.r_[np.zeros(equity_width, dtype=np.int64), np.ones(k, dtype=np.int64)],
            feature_prescaled=np.arange(equity_width + k) >= len(equity_features),
            edge_index=edge_index.T.copy(),
            edge_weight=np.ones(len(edges), dtype=np.float32),
            edge_type=edge_type,
            edge_type_names=EDGE_TYPES,
            edge_attr=np.eye(len(EDGE_TYPES), dtype=np.float32)[edge_type],
            edge_attr_names=np.array(EDGE_TYPES),
        )
