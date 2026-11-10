"""Causal rolling-ridge macro graph on the shared equity bar axis."""

import bisect
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from gnns.data.macro import FACTORS, load_macro
from gnns.data.snapshot import GraphSnapshot
from gnns.graphs.base import GraphBuilder

# Column kinds the builder scales itself from past data, so the Dataset leaves them as is:
# macro state is identical on every equity at t (ranking would make it all 0), interactions
# carry the size of the macro move, and the 5 macro nodes' context and one-hot identity
# columns mean nothing ranked across factors. Betas and magnitudes differ by stock and
# are ranked like the equity features.
PRESCALED = ("macro_level_", "macro_change_", "interaction_", "context_", "factor_")


def equity_macro_columns(level_factors: tuple[str, ...] = FACTORS) -> list[str]:
    """Equity-owned macro columns; `macro_level_*` only for `level_factors`."""
    return [
        f"{kind}_{f}"
        for kind in ("macro_level", "macro_change", "beta", "magnitude", "interaction")
        for f in (level_factors if kind == "macro_level" else FACTORS)
    ]


class MacroGraphBuilder(GraphBuilder):
    @classmethod
    def check_config(cls, cfg: dict) -> None:
        if "macro" not in cfg["graph"].get("sources", {}):
            raise ValueError("macro_sensitivity needs sources.macro in its graph file")

    @classmethod
    def load_sources(cls, root: Path, sources: dict) -> dict[str, tuple[pl.DataFrame, dict]]:
        return {"macro": load_macro(root, sources["macro"])}

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
        invert_factors: list[str] | tuple[str, ...] = ("rates",),
        level_factors: list[str] | tuple[str, ...] = ("oil", "gold"),
    ) -> None:
        if window_bars < 2 or not 2 <= min_obs <= window_bars or ridge <= 0:
            raise ValueError("invalid macro regression window, minimum or ridge")
        if (
            lag_bars < 0
            or max_stale_bars < 0
            or edge_mode not in {"magnitude", "signed", "constant", "shuffled"}
        ):
            raise ValueError("invalid lag, staleness or edge mode")
        if not {*invert_factors, *level_factors} <= set(FACTORS):
            raise ValueError(f"invert_factors and level_factors must be among {list(FACTORS)}")
        self.window_bars, self.min_obs, self.ridge = window_bars, min_obs, ridge
        self.lag_bars, self.max_stale_bars = lag_bars, max_stale_bars
        self.edge_mode, self.seed = edge_mode, seed
        # Factor order, not config order, so columns are stable.
        self.level_factors = tuple(f for f in FACTORS if f in level_factors)
        self._level = np.array([FACTORS.index(f) for f in self.level_factors], dtype=np.int64)
        self._sign = np.array([-1.0 if f in invert_factors else 1.0 for f in FACTORS])
        super().__init__(inputs, axis)

    def lookbacks(self) -> dict[str, int | None]:
        return {"prices": self.window_bars + self.lag_bars + 1, "features": 1, "macro": None}

    def build_at(self, t: datetime, inputs: dict[str, pl.DataFrame]) -> GraphSnapshot:
        feats = inputs["features"].sort("ticker")
        names = [c for c in feats.columns if c not in ("timestamp", "ticker")]
        macro_cols = equity_macro_columns(self.level_factors)
        node_names = (
            names
            + macro_cols
            + ["context_level", "context_change"]
            + [f"factor_{f}" for f in FACTORS]
        )
        eq_dim = len(names) + len(macro_cols)
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
        if np.any(levels[np.isfinite(levels)] <= 0):
            raise ValueError("macro prices must be positive for fractional changes")
        # Every factor is a price series; invert_factors flip the sign (rates: IEF falls
        # when yields rise, so minus its return keeps "yields up" positive).
        delta = (levels[1:] / levels[:-1] - 1) * self._sign
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
                h = history[np.isfinite(history).all(axis=1)][:, self._level]
                hsd = h.std(axis=0)
                if np.any(csd < 1e-12) or np.any(hsd < 1e-12):
                    continue
                lev = (current[self._level] - h.mean(axis=0)) / hsd
                standardized = (lev, change / csd)
                chosen.append(row)
                betas.append(beta)
        count = len(chosen)
        total = count + 5 if count else 0
        x = np.zeros((total, len(node_names)), dtype=np.float32)
        b = np.array(betas, dtype=np.float32).reshape(count, 5)
        if count:
            assert standardized is not None
            lev, chg = standardized
            for i, row in enumerate(chosen):
                x[i, :eq_dim] = np.r_[
                    [row[name] for name in names], lev, chg, b[i], np.abs(b[i]), b[i] * chg
                ]
            # Context nodes sorted by factor ID; explicit identity survives shared encoders.
            # Factors without a level get context_level 0; the one-hot tells them apart.
            context_level = np.zeros(5)
            context_level[self._level] = lev
            for j, f in enumerate(sorted(FACTORS)):
                k = FACTORS.index(f)
                x[count + j, eq_dim : eq_dim + 2] = [context_level[k], chg[k]]
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
            feature_prescaled=np.array([c.startswith(PRESCALED) for c in node_names]),
            edge_index=np.stack([src, dst]),
            edge_weight=np.abs(edge).reshape(-1).astype(np.float32),
            edge_type=np.tile(np.arange(5, dtype=np.int64), count),
            edge_type_names=FACTORS,
            edge_attr=edge.reshape(-1, 1).astype(np.float32),
            edge_attr_names=np.array(["beta"]),
        )
