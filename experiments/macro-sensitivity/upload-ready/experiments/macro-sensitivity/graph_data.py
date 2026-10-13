"""Causal graph sequences and train-only transforms for the fixed ETF pilot."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from macrograph import EQUITY, FACTORS, build_panel, read_series


@dataclass
class GraphPanel:
    dates: pd.DatetimeIndex
    assets: list
    equity: np.ndarray  # [date, asset, feature]
    macro: np.ndarray   # [date, factor, level/change]
    beta: np.ndarray    # signed [date, asset, factor]
    target: np.ndarray  # [date, asset]
    label_end: np.ndarray
    feature_valid: np.ndarray


def panel_to_graphs(panel, dates, assets):
    """Keep the actual session grid: missing dates never get squeezed out of sequences."""
    if panel.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/asset rows")
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "ticker"])
    ordered = panel.set_index(["date", "ticker"]).reindex(index)
    t, n, k = len(dates), len(assets), len(FACTORS)
    eq = ordered[EQUITY].to_numpy(float).reshape(t, n, -1)
    macro_cols = [f"{p}_{f}" for f in FACTORS for p in ["level", "change"]]
    macros = ordered[macro_cols].to_numpy(float).reshape(t, n, k, 2)
    if not np.allclose(macros, macros[:, :1], equal_nan=True):
        raise ValueError("Macro features differ between assets on the same date")
    macro = macros[:, 0]
    beta = ordered[[f"beta_{f}" for f in FACTORS]].to_numpy(float).reshape(t, n, k)
    target = ordered.target.to_numpy(float).reshape(t, n)
    end = pd.to_datetime(ordered.label_end).to_numpy().reshape(t, n)
    valid = np.isfinite(eq).all((1, 2)) & np.isfinite(macro).all((1, 2)) & np.isfinite(beta).all((1, 2))
    return GraphPanel(pd.DatetimeIndex(dates), assets, eq, macro, beta, target, end, valid)


def load_development(c):
    # Physically truncate raw inputs before feature construction: holdout never enters this runner.
    cutoff = pd.Timestamp(c["validation_end"])
    prices = pd.concat([read_series(t) for t in c["assets"]], axis=1, sort=True).loc[:cutoff]
    levels = pd.concat([read_series(f) for f in c["fred"]] +
                       [read_series(c["gold_proxy"]).rename("gold")], axis=1, sort=True).loc[:cutoff]
    levels = levels.reindex(levels.index.union(prices.index)).sort_index().ffill(limit=2).reindex(prices.index)
    panel = build_panel(prices, levels, c)
    return panel_to_graphs(panel, prices.index, c["assets"])


def eligible_ends(data, length):
    if length < 1:
        raise ValueError("Sequence length must be positive")
    complete = np.convolve(data.feature_valid.astype(int), np.ones(length, dtype=int), mode="full")[:len(data.dates)] == length
    labels = np.isfinite(data.target).all(axis=1) & ~pd.isna(data.label_end).any(axis=1)
    return np.flatnonzero(complete & labels)


def fold_indices(data, length, train_end, eval_end):
    ends = eligible_ends(data, length)
    boundary, final = np.datetime64(train_end), np.datetime64(eval_end)
    dates = data.dates.to_numpy()[ends]
    labels = data.label_end[ends].max(axis=1)
    train = ends[(dates <= boundary) & (labels <= boundary)]
    evaluate = ends[(dates > boundary) & (dates <= final) & (labels <= final)]
    if not len(train) or not len(evaluate):
        raise ValueError("Empty chronological fold")
    return train, evaluate


def edge_values(beta, mode, seed):
    """Controls computed once per calendar date, consistent across overlapping sequences."""
    if mode == "signed":
        return beta.copy()
    magnitude = np.abs(beta)
    if mode == "constant":
        return np.ones_like(beta)
    if mode == "shuffled":
        rng = np.random.default_rng(seed)
        out = magnitude.copy()
        for t in range(len(out)):
            out[t] = magnitude[t, rng.permutation(beta.shape[1])]
        return out
    if mode in ["magnitude", "frozen"]:
        return magnitude
    raise ValueError(mode)


def prepare_arrays(data, train, length, mode, seed):
    # Fit once on unique feature dates actually used by training sequences, not on validation.
    fit_dates = np.unique(np.concatenate([np.arange(t - length + 1, t + 1) for t in train]))
    stats = {}

    def standardize(name, array, axes):
        mean = array[fit_dates].mean(axis=axes, keepdims=True)
        sd = array[fit_dates].std(axis=axes, keepdims=True)
        sd = np.where(sd < 1e-8, 1., sd)
        stats[name] = {"mean": mean.tolist(), "scale": sd.tolist()}
        return (array - mean) / sd

    eq = standardize("equity", data.equity, (0, 1))
    macro = standardize("macro", data.macro, (0,))
    edge = edge_values(data.beta, mode, seed)
    # Positive scale only: preserve sign and preserve magnitude-only nonnegative attributes.
    scale = np.sqrt(np.mean(edge[fit_dates] ** 2, axis=(0, 1), keepdims=True))
    scale = np.where(scale < 1e-8, 1., scale)
    edge = edge / scale
    stats["edge_scale"] = scale.tolist()
    # Cross-sectional excess targets isolate ranking; raw returns remain available for spread metrics.
    target = data.target - data.target.mean(axis=1, keepdims=True)
    yscale = max(float(target[train].std()), 1e-8)
    stats["target_scale"] = yscale
    stats["fit_first"] = str(data.dates[fit_dates[0]].date())
    stats["fit_last"] = str(data.dates[fit_dates[-1]].date())
    return {"equity": eq.astype("float32"), "macro": macro.astype("float32"),
            "edge": edge.astype("float32"), "target": (target / yscale).astype("float32")}, stats


def sequence_batch(arrays, ends, length, frozen=False):
    ix = np.asarray(ends)[:, None] - np.arange(length - 1, -1, -1)[None, :]
    if (ix < 0).any():
        raise ValueError("Sequence would start before data")
    out = {key: arrays[key][ix].copy() for key in ["equity", "macro", "edge"]}
    if frozen:
        out["edge"][:] = out["edge"][:, :1]
    out["target"] = arrays["target"][ends]
    return out
