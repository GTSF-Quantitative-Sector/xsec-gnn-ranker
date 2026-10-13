"""Daily cross-sectional diagnostics; block intervals keep overlapping labels together."""
import numpy as np
import pandas as pd


def block_interval(frame, column, block=10, repeats=1000, seed=123):
    """Resample contiguous date blocks within each fold; seeds must be averaged first."""
    rng = np.random.default_rng(seed)
    groups = [g.sort_values("date")[column].to_numpy(float) for _, g in frame.groupby("fold")]
    if any(not np.isfinite(g).all() for g in groups):
        return float("nan"), float("nan")
    samples = []
    for _ in range(repeats):
        chunks = []
        for values in groups:
            length = min(block, len(values))
            starts = rng.integers(0, len(values) - length + 1, size=int(np.ceil(len(values) / length)))
            chunks.append(np.concatenate([values[s:s + length] for s in starts])[:len(values)])
        samples.append(np.concatenate(chunks).mean())
    return tuple(np.quantile(samples, [.025, .975]))


def summarize(metrics, block=10, repeats=1000):
    # Average each seed's metric, not its predictions. These are not ensemble portfolio results.
    daily = metrics.groupby(["model", "fold", "date"], as_index=False)[["rank_ic", "top3_minus_bottom3"]].mean()
    summaries = []
    for model, g in daily.groupby("model"):
        low, high = block_interval(g, "rank_ic", block, repeats)
        sl, sh = block_interval(g, "top3_minus_bottom3", block, repeats)
        per_seed = metrics[metrics.model == model].groupby("seed").rank_ic.mean()
        summaries.append({"model": model, "rank_ic": g.rank_ic.mean(), "ic_low": low, "ic_high": high,
                          "gross_spread": g.top3_minus_bottom3.mean(), "spread_low": sl, "spread_high": sh,
                          "seed_ic_std": per_seed.std(ddof=0), "dates": len(g), "seeds": len(per_seed)})
    pairs = [("gat_magnitude", "beta_mlp_magnitude"), ("gat_signed", "beta_mlp_signed"),
             ("temporal_magnitude", "gat_magnitude"), ("temporal_magnitude", "sequence_mlp"),
             ("gat_magnitude", "gat_constant"), ("gat_magnitude", "gat_shuffled"),
             ("temporal_magnitude", "temporal_constant"), ("temporal_magnitude", "temporal_shuffled"),
             ("temporal_magnitude", "temporal_frozen"), ("gat_signed", "gat_magnitude")]
    comparisons = []
    for a, b in pairs:
        left, right = daily[daily.model == a], daily[daily.model == b]
        if left.empty or right.empty:
            continue
        joined = left.merge(right, on=["fold", "date"], suffixes=("_a", "_b"), validate="one_to_one")
        joined["difference"] = joined.rank_ic_a - joined.rank_ic_b
        low, high = block_interval(joined, "difference", block, repeats)
        comparisons.append({"model": a, "reference": b, "ic_difference": joined.difference.mean(),
                            "low": low, "high": high, "dates": len(joined)})
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)
