"""Evaluation charts saved as PNGs. Each takes one table per model label (e.g. gatv2, mlp)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from gnns.eval.backtest import drawdown, net  # noqa: E402


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def ic_over_time(ics: dict[str, pl.DataFrame], rolling: int, path: Path) -> None:
    """Per-date rank IC of the first model (bars) and each model's rolling mean (lines)."""
    fig, ax = plt.subplots(figsize=(11, 4))
    first = next(iter(ics))
    ax.bar(ics[first]["timestamp"], ics[first]["ic"], width=4, alpha=0.3, label=f"{first} per date")
    for label, ic in ics.items():
        ax.plot(
            ic["timestamp"], ic["ic"].rolling_mean(rolling), label=f"{label} {rolling}-date mean"
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Rank IC over time (test rebalances)", ylabel="Spearman IC")
    ax.legend()
    _save(fig, path)


def cumulative_ic(ics: dict[str, pl.DataFrame], path: Path) -> None:
    """Running sum of per-date IC; a steady upward slope means a persistent signal."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, ic in ics.items():
        ax.plot(ic["timestamp"], ic["ic"].cum_sum(), label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Cumulative rank IC", ylabel="sum of per-date IC")
    ax.legend()
    _save(fig, path)


def ic_by_fold(fold_ics: dict[str, pl.DataFrame], path: Path) -> None:
    """Mean test IC per fold, models side by side: stability and graph lift."""
    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.8 / len(fold_ics)
    for i, (label, df) in enumerate(fold_ics.items()):
        ax.bar(
            df["fold"].to_numpy() + (i - (len(fold_ics) - 1) / 2) * width,
            df["mean_ic"],
            width=width,
            label=label,
        )
    folds = next(iter(fold_ics.values()))
    ax.set_xticks(folds["fold"].to_numpy(), folds["test_year"].to_list())
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Mean test IC by fold", xlabel="fold (test period start)", ylabel="mean IC")
    ax.legend()
    _save(fig, path)


def quantile_cascade(qrets: dict[str, pl.DataFrame], path: Path) -> None:
    """Mean forward return per score quantile (1 = top); a staircase means a real ranking."""
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(qrets)
    for i, (label, q) in enumerate(qrets.items()):
        means = q.group_by("quantile").agg(pl.col("ret").mean()).sort("quantile")
        x = means["quantile"].to_numpy() + (i - (len(qrets) - 1) / 2) * width
        ax.bar(x, means["ret"] * 100, width=width, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        title="Quantile cascade",
        xlabel="score quantile (1 = top)",
        ylabel="mean forward return per rebalance (%)",
    )
    ax.legend()
    _save(fig, path)


def quantile_curves(q: pl.DataFrame, label: str, path: Path) -> None:
    """Cumulative return of each score quantile over time."""
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = plt.get_cmap("RdYlGn_r")(np.linspace(0, 1, q["quantile"].max()))  # type: ignore[arg-type]
    for quantile in sorted(q["quantile"].unique()):
        part = q.filter(pl.col("quantile") == quantile).sort("timestamp")
        ax.plot(
            part["timestamp"],
            (1 + part["ret"]).cum_prod(),
            label=f"Q{quantile}",
            color=colors[int(quantile) - 1],
        )
    ax.set(title=f"Quantile equity curves ({label})", ylabel="growth of $1", yscale="log")
    ax.legend(ncol=2)
    _save(fig, path)


def score_spread(preds: dict[str, pl.DataFrame], path: Path) -> None:
    """Per-date gap between the highest and lowest score. Near zero means the model gives
    every node about the same score, so its ranking is noise."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, p in preds.items():
        s = p.group_by("timestamp").agg(spread=pl.col("score").max() - pl.col("score").min())
        s = s.sort("timestamp")
        ax.plot(s["timestamp"], s["spread"], label=f"{label} (median {s['spread'].median():.2g})")
    ax.set(title="Score spread per date (max - min)", ylabel="spread", yscale="log")
    ax.legend()
    _save(fig, path)


def feature_importance(importances: dict[str, list[dict[str, float]]], path: Path) -> None:
    """Permutation importance per input (and "edges"), per model: bar = mean over folds,
    error bar = spread (std) across folds. Above zero: the model relies on it and it helps
    on validation dates; around zero: unused; below zero: it hurts out of sample."""
    keys = list(next(iter(importances.values()))[0])
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(keys)), 4))
    width = 0.8 / len(importances)
    for i, (label, folds) in enumerate(importances.items()):
        vals = np.array([[f[k] for k in keys] for f in folds])
        x = np.arange(len(keys)) + (i - (len(importances) - 1) / 2) * width
        ax.bar(x, vals.mean(0), width=width, yerr=vals.std(0), capsize=3, label=label)
    ax.set_xticks(np.arange(len(keys)), keys, rotation=20)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        title="Permutation importance (validation IC drop when the input is scrambled)",
        ylabel="IC drop",
    )
    ax.legend()
    _save(fig, path)


def feature_correlation(corr: np.ndarray, names: list[str], path: Path) -> None:
    """Average within-date rank correlation between features: high |value| means the two
    inputs overlap, so permute or ablate them as a group."""
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2.5, 1.0 * len(names) + 1.8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    ax.set_title("Feature correlation (within-date rank, search period)")
    _save(fig, path)


def training_curves(histories: dict[str, list[list[dict]]], path: Path) -> None:
    """Per fold: validation IC (solid), training loss (dashed), best epoch, and the loss of
    giving every node the same score (grey dotted): a loss on that line learned nothing."""
    n = len(next(iter(histories.values())))
    cols = min(n, 5)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    for f in range(n):
        ax = axes[f // cols][f % cols]
        loss_ax = ax.twinx()
        for i, (label, folds) in enumerate(histories.items()):
            h = pl.DataFrame(folds[f])
            color = f"C{i}"
            ax.plot(h["epoch"], h["val_ic"], color=color, label=f"{label} val IC")
            loss_ax.plot(h["epoch"], h["train_loss"], color=color, ls="--", alpha=0.5)
            best = h.filter(pl.col("val_ic") == pl.col("val_ic").max())["epoch"][0]
            ax.axvline(best, color=color, lw=0.8, alpha=0.6)
        first = next(iter(histories.values()))[f]
        ref = next((e["no_info_loss"] for e in first if "no_info_loss" in e), None)
        if ref is not None:  # runs before this reference was logged have none
            loss_ax.axhline(ref, color="grey", ls=":", lw=1.2)
        ax.set_title(f"fold {f + 1}", fontsize=9)
        ax.tick_params(labelsize=7)
        loss_ax.tick_params(labelsize=7)
    for f in range(n, rows * cols):
        axes[f // cols][f % cols].axis("off")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(
        "Training curves: val IC (solid, left), train loss (dashed, right), best epoch, "
        "no-information loss (grey dotted)"
    )
    _save(fig, path)


def backtest_equity(
    rebalances: dict[str, pl.DataFrame], cost_bps: list[float], benchmark: str, path: Path
) -> None:
    """Growth of $1 per model: top-k net of each cost level, equal-weight universe, benchmark."""
    fig, axes = plt.subplots(len(rebalances), 1, figsize=(11, 3.5 * len(rebalances)), squeeze=False)
    for ax, (label, p) in zip(axes[:, 0], rebalances.items(), strict=True):
        t = p["timestamp"]
        for bps in cost_bps:
            ax.plot(t, (1 + p.select(net(bps)).to_series()).cum_prod(), label=f"top-k, {bps:g} bps")
        ax.plot(t, (1 + p["universe"]).cum_prod(), color="grey", label="equal-weight universe")
        ax.plot(t, (1 + p["benchmark"]).cum_prod(), color="black", label=benchmark)
        ax.set(title=f"{label}: top-k backtest", ylabel="growth of $1", yscale="log")
        ax.legend(fontsize=8)
    _save(fig, path)


def backtest_drawdown(
    rebalances: dict[str, pl.DataFrame], cost_bps: list[float], path: Path
) -> None:
    """How far each model's portfolio sits below its running peak, per cost level."""
    fig, axes = plt.subplots(len(rebalances), 1, figsize=(11, 3 * len(rebalances)), squeeze=False)
    for ax, (label, p) in zip(axes[:, 0], rebalances.items(), strict=True):
        for bps in cost_bps:
            r = p.select(net(bps)).to_series().to_numpy()
            ax.plot(p["timestamp"], drawdown(r) * 100, label=f"{bps:g} bps")
        ax.set(title=f"{label}: drawdown", ylabel="% below peak")
        ax.legend(fontsize=8)
    _save(fig, path)


def backtest_per_year(years: dict[str, pl.DataFrame], held: str, path: Path) -> None:
    """Net return and Sharpe per calendar year for each model and cost level: is the result
    consistent across years, or carried by a few? Ticks show each year's rebalance count
    (n), so partial years (e.g. a fold starting in late December) are visible; `held`
    says how long one rebalance is held."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    width = 0.8 / len(years)
    for i, (name, y) in enumerate(years.items()):
        x = y["year"].to_numpy() + (i - (len(years) - 1) / 2) * width
        top.bar(x, y["net_return"] * 100, width=width, label=name)
        bottom.bar(x, y["sharpe"], width=width, label=name)
    for ax in (top, bottom):
        ax.axhline(0, color="black", lw=0.8)
    first = next(iter(years.values()))
    ticks = [f"{y}\nn={n}" for y, n in first.select("year", "rebalances").iter_rows()]
    bottom.set_xticks(first["year"].to_list(), ticks)
    top.set(title="Backtest by calendar year", ylabel="net return (%)")
    bottom.set(xlabel=f"year of rebalance date (n = rebalances that year, {held})", ylabel="Sharpe")
    top.legend(fontsize=8, ncol=3)
    _save(fig, path)
