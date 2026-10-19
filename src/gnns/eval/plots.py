"""Evaluation charts saved as PNGs. Each takes one table per model label (e.g. gatv2, mlp)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402


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
    ax.set(title="Rank IC over time (test periods)", ylabel="Spearman IC")
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
        ylabel="mean forward return per period (%)",
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


def equity_curves(series: dict[str, pl.DataFrame], path: Path) -> None:
    """Growth of $1 for each return series (top-k portfolios, equal-weight universe, SPY)."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, r in series.items():
        ax.plot(r["timestamp"], (1 + r["ret"]).cum_prod(), label=label)
    ax.set(title="Top-k equity curves (no costs)", ylabel="growth of $1", yscale="log")
    ax.legend()
    _save(fig, path)


def training_curves(histories: dict[str, list[list[dict]]], path: Path) -> None:
    """Per fold: validation IC (solid) and training loss (dashed), best epoch marked."""
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
        ax.set_title(f"fold {f + 1}", fontsize=9)
        ax.tick_params(labelsize=7)
        loss_ax.tick_params(labelsize=7)
    for f in range(n, rows * cols):
        axes[f // cols][f % cols].axis("off")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Training curves: val IC (solid, left), train loss (dashed, right), best epoch")
    _save(fig, path)
