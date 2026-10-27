# GATv2 and temporal macro graph: development findings

Implemented and evaluated October 12, 2026, using the existing September data cache. This supersedes the implementation status in the earlier ridge-only findings, not its historical results.

## Completed implementation

- A dense, directed macro-to-ETF GATv2 with factor-specific node encoders, beta edge attributes, ETF residual features, and ETF-only scores.
- A temporal GATv2+GRU adaptation using 20 historical snapshots. It is not an exact implementation of the published THGNN model.
- Nonlinear MLP and non-graph sequence baselines; signed, magnitude, constant, shuffled, and frozen-beta controls.
- Two expanding development folds, three random seeds, fixed 30-epoch training, saved checkpoints/preprocessing, block intervals, paired comparisons, and attention summaries across VIX/rate regimes.

The complete run trained 84 models (14 variants × 2 folds × 3 seeds). Fold 1 had 213 training dates and fold 2 had 277; each had 58 evaluation dates, for 116 distinct evaluation dates. The later holdout was excluded before graph feature construction. Eighteen automated tests passed, including temporal causality, graph batch isolation, edge gradients, and exclusion of raw holdout values.

## Selected results

| Model | Mean Rank IC | Gross five-session top-three minus bottom-three spread |
|---|---:|---:|
| Equity-only MLP | 0.0406 | 0.215% |
| Macro MLP | 0.0515 | 0.247% |
| Magnitude-beta MLP | 0.1042 | 0.379% |
| GATv2, magnitude edges | 0.0968 | 0.351% |
| GATv2, constant edges | 0.0855 | 0.324% |
| GATv2, shuffled edges | 0.0913 | 0.365% |
| Non-graph MLP+GRU | 0.0811 | 0.315% |
| Temporal graph, magnitude edges | 0.0578 | 0.226% |

Metrics are averaged across seed-specific results, not predictions from an ensemble. Forward returns overlap and are gross of costs; these spreads are not daily tradable portfolio returns. The samples and models differ from the earlier ridge-only run, so its numbers should not be directly compared with this table.

## Interpretation

The magnitude-beta MLP had the highest mean Rank IC in this configuration. GATv2 did not improve on it: the paired IC difference was -0.0073, with a conditional 95% block interval of [-0.0467, 0.0474]. The temporal graph also did not improve on the snapshot graph: difference -0.0391, interval [-0.0759, 0.0193].

Estimated edges had only small differences from controls. GATv2 magnitude minus constant edges was +0.0113 (interval [-0.0073, 0.0210]); magnitude minus shuffled edges was +0.0055 (interval [-0.0116, 0.0130]). These intervals include zero. The experiment therefore does not yet establish that the estimated graph structure or temporal extension adds predictive value.

Attention exports and regime tables are implemented. The overall attention plot puts relatively more weight on gold and credit than oil in this run, including in control graphs. This describes learned attention allocation, not causal importance; it does not establish the economic correctness of the learned relationships.

## Limits and next research

The study remains a short-history ETF pilot with latest-vintage macro data, assumed publication lags, and limited credit history. Intervals are conditional on fitted models and are not adjusted for multiple comparisons. There is no direct control for firms' disclosed exposures, supply-chain model comparison, or cost-adjusted portfolio simulation.

The next priority is stronger data and validation, not a claim that temporal GNNs outperform. Obtain longer credit history and verified timing, repeat on broader regimes, test factor removal and beta-window stability, and freeze the protocol before any final holdout evaluation.

## Reproduce

See [README](../README.md) for environment setup and [graph protocol](graph_protocol.md) for architecture and methodology. Run `python train_graphs.py` from the experiment directory. The completed local run is in `outputs/graphs/development_graph_comparison/`; generated artifacts are intentionally excluded from Git. Fresh downloads can change the available credit history, so exact reproduction requires the recorded cached inputs and environment versions.
