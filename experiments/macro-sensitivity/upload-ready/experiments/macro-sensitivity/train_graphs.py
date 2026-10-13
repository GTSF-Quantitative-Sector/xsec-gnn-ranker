"""Run fixed-budget development comparisons. Intentionally has no holdout option."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric

from macrograph import ROOT, FACTORS, config, daily_metrics
from graph_data import load_development, fold_indices, prepare_arrays, sequence_batch
from graph_evaluation import summarize
from graph_models import make_model, edge_mode


def tensors(batch):
    return {k: torch.from_numpy(v) for k, v in batch.items()}


def train_one(model, arrays, ends, settings, mode, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    rng = np.random.default_rng(seed)
    history = []
    model.train()
    for epoch in range(settings["epochs"]):
        order = rng.permutation(ends)
        total = 0.
        for start in range(0, len(order), settings["batch_size"]):
            batch_ends = order[start:start + settings["batch_size"]]
            b = tensors(sequence_batch(arrays, batch_ends, settings["sequence_length"], mode == "frozen"))
            optimizer.zero_grad()
            pred, _ = model(b["equity"], b["macro"], b["edge"])
            pred = pred - pred.mean(dim=1, keepdim=True)
            loss = ((pred - b["target"]) ** 2).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(batch_ends)
        history.append({"epoch": epoch + 1, "train_mse": total / len(ends)})
    return history


def predict(model, arrays, ends, settings, mode):
    predictions, attentions = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ends), settings["batch_size"]):
            b = tensors(sequence_batch(arrays, ends[start:start + settings["batch_size"]], settings["sequence_length"], mode == "frozen"))
            pred, alpha = model(b["equity"], b["macro"], b["edge"])
            predictions.append(pred.numpy())
            if alpha is not None:
                attentions.append(alpha.numpy())
    return np.concatenate(predictions), np.concatenate(attentions) if attentions else None


def regime_frame(data, train, evaluate):
    vix_index = FACTORS.index("vix")
    threshold = np.median(data.macro[train, vix_index, 0])
    rate = pd.Series(data.macro[:, FACTORS.index("rates"), 0]).diff(21).to_numpy()
    frame = pd.DataFrame({"date": data.dates[evaluate],
                          "vix_regime": np.where(data.macro[evaluate, vix_index, 0] > threshold, "high_vix", "low_vix"),
                          "rate_regime": np.where(rate[evaluate] > 0, "rising_rates", "flat_or_falling_rates")})
    return frame, float(threshold)


def write_report(out, summary, comparisons, coverage, settings, smoke):
    table = "| Model | Rank IC | 95% block interval | Gross 5-session spread |\n|---|---:|---:|---:|\n"
    for row in summary.itertuples():
        table += f"| {row.model} | {row.rank_ic:.4f} | [{row.ic_low:.4f}, {row.ic_high:.4f}] | {100 * row.gross_spread:.3f}% |\n"
    text = "# " + ("SMOKE TEST — " if smoke else "") + "Macro graph development comparison\n\n"
    text += "This evaluates GATv2 and a factor-typed GATv2+GRU adaptation, not an exact THGNN reproduction. No holdout observations were loaded into feature construction.\n\n"
    text += table
    text += "\nMetrics average separately trained seeds on identical dates. These are overlapping gross forward-return spreads, not tradable daily returns. Intervals resample ten-session date blocks within each fold (or the block size in run.json), conditional on fitted models; they do not correct for searching across models.\n\n"
    text += "## Paired Rank IC differences\n\n"
    if len(comparisons):
        text += "| Model minus reference | Difference | 95% block interval |\n|---|---:|---:|\n"
        for row in comparisons.itertuples():
            text += f"| {row.model} − {row.reference} | {row.ic_difference:.4f} | [{row.low:.4f}, {row.high:.4f}] |\n"
    text += "\n## Protocol and limitations\n\n"
    text += f"Fixed {settings['epochs']} epochs, seeds {settings['seeds']}, {settings['sequence_length']}-session sequences. No validation-based epoch selection. Both folds are development data previously used for research. All models use complete 11-ETF sequences and the same eligible endpoint dates.\n\n"
    for row in coverage:
        text += f"- {row['fold']}: {row['train_dates']} training dates ({row['train_first']}–{row['train_last']}); {row['eval_dates']} evaluation dates ({row['eval_first']}–{row['eval_last']}).\n"
    text += "\nLatest-vintage data, assumed macro publication delays, short credit coverage, and ETF proxies remain limitations. No disclosed-exposure control, supply-chain graph comparison, or cost-adjusted portfolio simulation is implemented. Magnitude-only models never receive beta signs.\n\n"
    text += "Attention is measured at the last snapshot, normalized across macro factors for each ETF, then averaged over heads. It is descriptive, not causal importance or a decomposition of temporal model predictions. Regimes use lagged observed VIX (training-median threshold) and the sign of the lagged yield's 21-session change.\n\n"
    text += "Artifacts: summary.csv, paired_comparisons.csv, predictions.csv, daily_metrics.csv, attention.csv, attention_by_regime.csv, metrics_by_regime.csv, training_history.csv, per-model checkpoints/scalers, run.json, and diagnostic PNGs.\n"
    (out / "REPORT.md").write_text(text, encoding="utf-8")


def plots(out, summary, attention, data, development_ends):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ys = np.arange(len(summary))
    # Plot interval endpoints directly; a bootstrap interval need not contain its point estimate.
    ax.hlines(ys, summary.ic_low, summary.ic_high, color="#718096")
    ax.scatter(summary.rank_ic, ys, color="#2563eb", zorder=3)
    ax.set_yticks(ys, summary.model)
    ax.axvline(0, color="black", linewidth=.7)
    ax.set_xlabel("Mean daily Rank IC (seed-averaged); 95% block interval")
    ax.set_title("Development data only — short-history ETF pilot")
    fig.tight_layout()
    fig.savefig(out / "rank_ic.png", dpi=160)
    plt.close(fig)
    if len(attention):
        table = attention.groupby(["model", "factor"]).attention.mean().unstack().reindex(columns=FACTORS)
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(table.to_numpy(), aspect="auto", vmin=0)
        ax.set_xticks(range(len(FACTORS)), FACTORS)
        ax.set_yticks(range(len(table)), table.index)
        ax.set_title("Mean last-snapshot attention — descriptive, not causal")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / "attention.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(len(FACTORS), 1, figsize=(10, 10), sharex=True)
    for k, ax in enumerate(axes):
        for i, ticker in enumerate(data.assets):
            ax.plot(data.dates[development_ends], data.beta[development_ends, i, k], label=ticker, linewidth=.7)
        ax.set_ylabel(FACTORS[k])
        ax.axhline(0, color="black", linewidth=.5)
    axes[0].set_title("Lagged signed rolling betas per one trailing factor SD")
    axes[0].legend(ncol=6, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "betas.png", dpi=160)
    plt.close(fig)


def run(settings, out, smoke=False):
    c = config()
    for fold in settings["folds"]:
        if pd.Timestamp(fold["eval_end"]) > pd.Timestamp(c["validation_end"]):
            raise ValueError("Fold exceeds development cutoff; holdout is locked")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = load_development(c)
    coverage, histories, predictions, metrics, attentions = [], [], [], [], []
    all_ends = []
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in list((ROOT / "data" / "raw").glob("*.csv")) +
              [ROOT / f for f in ["macrograph.py", "graph_data.py", "graph_models.py", "graph_evaluation.py", "train_graphs.py"]]}
    metadata = {"data_config": c, "experiment": settings, "smoke": smoke,
                "started_utc": datetime.now(timezone.utc).isoformat(), "holdout_loaded": False,
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "torch_geometric": torch_geometric.__version__, "numpy": np.__version__, "pandas": pd.__version__},
                "hashes": hashes, "coverage": coverage, "status": "running"}
    (out / "run.json").write_text(json.dumps(metadata, indent=2))
    for fold in settings["folds"]:
        train, evaluate = fold_indices(data, settings["sequence_length"], fold["train_end"], fold["eval_end"])
        if len(train) < 100 or len(evaluate) < 20:
            raise ValueError("Insufficient fold history")
        all_ends.extend(train)
        all_ends.extend(evaluate)
        regimes, vix_threshold = regime_frame(data, train, evaluate)
        coverage.append({"fold": fold["name"], "train_dates": len(train), "eval_dates": len(evaluate),
                         "train_first": str(data.dates[train[0]].date()), "train_last": str(data.dates[train[-1]].date()),
                         "eval_first": str(data.dates[evaluate[0]].date()), "eval_last": str(data.dates[evaluate[-1]].date()),
                         "vix_training_median": vix_threshold})
        for name in settings["models"]:
            for seed in settings["seeds"]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                mode = edge_mode(name)
                arrays, stats = prepare_arrays(data, train, settings["sequence_length"], mode, seed)
                model = make_model(name, len(data.assets), settings["hidden_dim"], settings["heads"])
                history = train_one(model, arrays, train, settings, mode, seed)
                pred, alpha = predict(model, arrays, evaluate, settings, mode)
                key = {"fold": fold["name"], "model": name, "seed": seed}
                histories.extend([{**key, **h} for h in history])
                scored = pd.DataFrame({"date": np.repeat(data.dates[evaluate], len(data.assets)),
                                       "ticker": np.tile(data.assets, len(evaluate)),
                                       "target": data.target[evaluate].reshape(-1),
                                       "prediction": pred.reshape(-1) * stats["target_scale"]})
                daily = daily_metrics(scored).merge(regimes, on="date", validate="one_to_one")
                predictions.append(scored.assign(**key))
                metrics.append(daily.assign(**key))
                if alpha is not None:
                    att = pd.DataFrame({"date": np.repeat(data.dates[evaluate], len(data.assets) * len(FACTORS)),
                                        "ticker": np.tile(np.repeat(data.assets, len(FACTORS)), len(evaluate)),
                                        "factor": np.tile(FACTORS, len(evaluate) * len(data.assets)),
                                        "attention": alpha.reshape(-1),
                                        "beta_signed_metadata": data.beta[evaluate].reshape(-1)})
                    attentions.append(att.merge(regimes, on="date", validate="many_to_one").assign(**key))
                artifact = out / "checkpoints" / fold["name"] / f"{name}_{seed}"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), artifact.with_suffix(".pt"))
                artifact.with_suffix(".json").write_text(json.dumps({**key, "stats": stats,
                    "parameters": sum(p.numel() for p in model.parameters()), "settings": settings,
                    "asset_order": data.assets, "factor_order": FACTORS}, indent=2))
                print(f"{fold['name']} {name} seed={seed}: IC={daily.rank_ic.mean():.4f}, loss={history[-1]['train_mse']:.4f}", flush=True)
    metrics = pd.concat(metrics, ignore_index=True)
    attention = pd.concat(attentions, ignore_index=True) if attentions else pd.DataFrame()
    summary, comparisons = summarize(metrics, settings["bootstrap_block"], settings["bootstrap_repeats"])
    for name, frame in [("summary", summary), ("paired_comparisons", comparisons), ("daily_metrics", metrics),
                        ("predictions", pd.concat(predictions, ignore_index=True)), ("attention", attention),
                        ("training_history", pd.DataFrame(histories))]:
        frame.to_csv(out / f"{name}.csv", index=False)
    regime_daily = metrics.groupby(["model", "fold", "date", "vix_regime", "rate_regime"], as_index=False)[["rank_ic", "top3_minus_bottom3"]].mean()
    regime_daily.groupby(["model", "vix_regime", "rate_regime"]).agg(rank_ic=("rank_ic", "mean"),
        gross_spread=("top3_minus_bottom3", "mean"), dates=("date", "size")).to_csv(out / "metrics_by_regime.csv")
    if len(attention):
        attention.groupby(["model", "vix_regime", "rate_regime", "factor"]).attention.mean().to_csv(out / "attention_by_regime.csv")
    write_report(out, summary, comparisons, coverage, settings, smoke)
    plots(out, summary, attention, data, np.unique(all_ends))
    metadata["status"] = "complete"
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "run.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(f"Report: {out / 'REPORT.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Two-epoch plumbing check, not research evidence")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = json.loads((ROOT / "graph_config.json").read_text())
    if args.smoke:
        settings.update(epochs=2, seeds=[7], models=["beta_mlp_magnitude", "gat_magnitude", "temporal_magnitude"], bootstrap_repeats=100)
        settings["folds"] = settings["folds"][:1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output or ROOT / "outputs" / "graphs" / (("smoke_" if args.smoke else "development_") + stamp)
    run(settings, out, args.smoke)


if __name__ == "__main__":
    main()
