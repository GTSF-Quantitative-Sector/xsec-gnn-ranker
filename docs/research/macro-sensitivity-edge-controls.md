# Macro-sensitivity graph: development edge controls

November 10, 2026. Does supplying rolling macro beta magnitudes as edge inputs improve sector ETF return rankings? In this development comparison, **constant edge inputs ranked better than the estimated beta inputs in all three seeds and seven of nine folds**.

| Model | Mean Rank IC | Annualized top-three return, 3 bps per trade side |
|---|---:|---:|
| GATv2, estimated beta magnitudes | 0.0158 | 8.82% |
| GATv2, constant edge values | **0.0404** | **10.80%** |
| GATv2, shuffled beta magnitudes | 0.0191 | 8.77% |
| Matched MLP | 0.0227 | 8.23% |

Metrics are averaged across seeds, not ensemble results. SPY returned 11.64% annualized over the same windows before benchmark costs. The constant-edge variant had the strongest average model result but still trailed that reference after modeled costs.

**Setup.** Runs used the Massive data-builder trial at `33cbf99` with the sector ETF universe, five-session next-open returns, nine walk-forward development folds, and seeds 7, 19 and 43. The macro inputs are traded proxies, including HYG for credit, so this is not a replication of the [earlier pilot](macro-sensitivity-pilot/graph_findings.md). All variants used identical ETF/date/return rows on 453 ranking dates per seed. A local cutoff excluded labels that crossed into 2024; the last scored signal was December 20, 2023, and the last exit was December 29, 2023. The shared holdout remains unscored.

**What changed between models.** The constant control replaces both edge input channels with ones. The shuffled control permutes complete equity beta vectors within each date. Both retain the same directed macro-to-equity links and leave beta features on equity nodes. The matched MLP receives the same equity-owned macro information and does not use edges; its scores matched exactly across control runs for each seed.

**Paired differences.** Estimated minus constant was −0.0246 Rank IC (conditional 95% block interval: −0.0399 to −0.0086). Estimated minus shuffled was −0.0034 (−0.0176 to +0.0121). Estimated minus matched MLP was −0.0070 (−0.0275 to +0.0140). Intervals use 4,000 circular block resamples of seed-averaged date differences within each fold, with ten rebalances per block. Four-rebalance blocks gave the same qualitative conclusions. The intervals are conditional on fitted models, without multiple-comparison adjustment; the seeds reuse the same market dates.

**Next experiment.** Compare equity-only, macro-state, and macro-plus-beta MLPs on the same development folds to test whether beta features help as inputs. The edge controls only address the value of supplying estimated betas on the edges. Agree on the final confirmation period with the team before scoring it, given that the earlier standalone pilot evaluated 2025.
