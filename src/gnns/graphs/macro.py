"""Causal rolling-ridge macro graph on the shared equity bar axis."""

import bisect
from datetime import datetime

import numpy as np
import polars as pl

from gnns.data.macro import FACTORS
from gnns.data.snapshot import GraphSnapshot
from gnns.graphs.base import GraphBuilder


def equity_macro_columns() -> list[str]:
    return [
        f"{kind}_{f}"
        for kind in ("macro_level", "macro_change", "beta", "magnitude", "interaction")
        for f in FACTORS
    ]


class MacroGraphBuilder(GraphBuilder):
    def __init__(
        self,
        inputs: dict[str, pl.DataFrame],
        axis: pl.Series,
        *,
        window_bars: int = 252,
        min_obs: int = 200,
        ridge: float = 10.0,
        lag_bars: int = 0,
        max_stale_bars: int = 2,
        edge_mode: str = "magnitude",
        seed: int = 7,
    ) -> None:
        if window_bars < 2 or not 2 <= min_obs <= window_bars or ridge <= 0:
            raise ValueError("invalid macro regression window, minimum or ridge")
        if (
            lag_bars < 0
            or max_stale_bars < 0
            or edge_mode not in {"magnitude", "signed", "constant", "shuffled"}
        ):
            raise ValueError("invalid lag, staleness or edge mode")
        self.window_bars, self.min_obs, self.ridge = window_bars, min_obs, ridge
        self.lag_bars, self.max_stale_bars = lag_bars, max_stale_bars
        self.edge_mode, self.seed = edge_mode, seed
        super().__init__(inputs, axis)

    def lookbacks(self) -> dict[str, int | None]:
        return {"prices": self.window_bars + self.lag_bars + 1, "features": 1, "macro": None}

    def build_at(self, t: datetime, inputs: dict[str, pl.DataFrame]) -> GraphSnapshot:
        feats = inputs["features"].sort("ticker")
        names = [c for c in feats.columns if c not in ("timestamp", "ticker")]
        node_names = (
            names
            + equity_macro_columns()
            + ["context_level", "context_change"]
            + [f"factor_{f}" for f in FACTORS]
        )
        eq_dim = len(names) + 25
        axis = self._axis.filter(self._axis <= t).tail(self.window_bars + self.lag_bars + 1)
        n = len(axis)
        levels = np.full((n, 5), np.nan)
        # Map known-at events to the first bar at or after publication. Rows later
        # than t have already been excluded by GraphBuilder.build.
        for k, factor in enumerate(FACTORS):
            events = inputs["macro"].filter(pl.col("factor") == factor)
            if events.is_empty():
                continue
            event_ts = events["timestamp"].to_list()
            values = events["value"].to_numpy()
            for j, stamp in enumerate(axis):
                pos = bisect.bisect_right(event_ts, stamp) - 1
                if pos >= 0:
                    # Count staleness on the full universe axis, including history
                    # before this regression window.
                    age = self._axis.search_sorted(stamp) - self._axis.search_sorted(event_ts[pos])
                    if age <= self.max_stale_bars:
                        levels[j, k] = values[pos]
        delta = np.diff(levels, axis=0)
        delta[:, :2] *= 100
        for k in (3, 4):
            if np.any(levels[:, k][np.isfinite(levels[:, k])] <= 0):
                raise ValueError("oil and gold must be positive for fractional changes")
            delta[:, k] = levels[1:, k] / levels[:-1, k] - 1
        stop = n - self.lag_bars - 1
        lo = max(0, stop - self.window_bars)
        xreg = delta[lo:stop]
        current = levels[stop] if stop >= 0 else np.full(5, np.nan)
        change = delta[stop - 1] if stop > 0 else np.full(5, np.nan)
        history = levels[lo + 1 : stop + 1]
        chosen, betas, standardized = [], [], None
        if (
            len(xreg) >= self.window_bars
            and np.isfinite(current).all()
            and np.isfinite(change).all()
        ):
            for row in feats.iter_rows(named=True):
                prices = dict(
                    inputs["prices"]
                    .filter(pl.col("ticker") == row["ticker"])
                    .select("timestamp", "close")
                    .iter_rows()
                )
                close = np.array([prices.get(stamp, np.nan) for stamp in axis])
                y = (close[1:] / close[:-1] - 1)[lo:stop]
                good = np.isfinite(xreg).all(axis=1) & np.isfinite(y)
                if good.sum() < self.min_obs:
                    continue
                xx, yy = xreg[good], y[good]
                sd = xx.std(axis=0)
                if np.any(sd < 1e-12):
                    continue
                z = (xx - xx.mean(axis=0)) / sd
                beta = np.linalg.solve(z.T @ z + self.ridge * np.eye(5), z.T @ (yy - yy.mean()))
                # Macro scaling uses past available macro data only, common across equities.
                common = np.isfinite(xreg).all(axis=1)
                csd = xreg[common].std(axis=0)
                h = history[np.isfinite(history).all(axis=1)]
                hsd = h.std(axis=0)
                if np.any(csd < 1e-12) or np.any(hsd < 1e-12):
                    continue
                standardized = ((current - h.mean(axis=0)) / hsd, change / csd)
                chosen.append(row)
                betas.append(beta)
        count = len(chosen)
        total = count + 5 if count else 0
        x = np.zeros((total, len(node_names)), dtype=np.float32)
        b = np.array(betas, dtype=np.float32).reshape(count, 5)
        if count:
            lev, chg = standardized
            for i, row in enumerate(chosen):
                x[i, :eq_dim] = np.r_[
                    [row[name] for name in names], lev, chg, b[i], np.abs(b[i]), b[i] * chg
                ]
            # Context nodes sorted by factor ID; explicit identity survives shared encoders.
            for j, f in enumerate(sorted(FACTORS)):
                k = FACTORS.index(f)
                x[count + j, eq_dim : eq_dim + 2] = [lev[k], chg[k]]
                x[count + j, eq_dim + 2 + k] = 1
        src = np.array(
            [count + sorted(FACTORS).index(f) for _ in chosen for f in FACTORS], dtype=np.int64
        )
        dst = np.repeat(np.arange(count, dtype=np.int64), 5)
        edge = b.copy() if self.edge_mode == "signed" else np.abs(b)
        if self.edge_mode == "constant":
            edge = np.ones_like(b)
        elif self.edge_mode == "shuffled":
            # Date-local deterministic permutation; stable under future-data deletion.
            rng = np.random.default_rng([self.seed, t.year, t.month, t.day])
            edge = edge[rng.permutation(count)]
        return GraphSnapshot(
            timestamp=t,
            node_ids=np.array(
                [r["ticker"] for r in chosen]
                + ([f"macro:{f}" for f in sorted(FACTORS)] if count else []),
                dtype=str,
            ),
            node_type=np.r_[
                np.zeros(count, dtype=np.int64), np.ones(total - count, dtype=np.int64)
            ],
            node_type_names=("equity", "macro"),
            target_mask=np.arange(total) < count,
            x=x,
            feature_names=np.array(node_names),
            feature_owner=np.r_[
                np.zeros(eq_dim, dtype=np.int64), np.ones(len(node_names) - eq_dim, dtype=np.int64)
            ],
            edge_index=np.stack([src, dst]),
            edge_weight=np.abs(edge).reshape(-1).astype(np.float32),
            edge_type=np.tile(np.arange(5, dtype=np.int64), count),
            edge_type_names=FACTORS,
            edge_attr=edge.reshape(-1, 1).astype(np.float32),
            edge_attr_names=np.array(["beta"]),
        )
