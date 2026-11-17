# MLP feature comparison — November 17, 2026

Question: do macro state and beta-derived node features improve return rankings without graph message passing?

All three variants use the same 34 input slots, shared two-layer MLP, initialization seed, training/validation rules, and nine development folds. Excluded feature columns are set to zero after the shared normalization. Equity-only retains the five momentum/volatility columns; macro-state adds the seven macro level/change columns; macro-plus-beta restores signed betas, magnitudes and interactions. Context-only columns are zero on equities. This controls nominal network size but does not guarantee the same effective complexity or optimization behavior.

The 81 fits cover three variants, three seeds (7, 19, 43), and nine folds. Every variant scores the same ETF/date/return rows and 453 ranking dates per seed. All scored return labels end before January 1, 2024. The full-feature reruns reproduce prior MLP scores exactly (maximum absolute difference zero). Data is the same frozen Massive proxy dataset used in the [edge-control study](macro-sensitivity-edge-controls.md), trial commit 33cbf99; no new data download or holdout evaluation was performed.

| MLP inputs | Mean Rank IC | Annualized top-three return at 3 bps per side | Mean Sharpe |
|---|---:|---:|---:|
| Equity-only | 0.0008 | 7.43% | 0.43 |
| Equity + macro state | -0.0044 | 5.76% | 0.33 |
| Equity + macro state + betas/interactions | **0.0227** | **8.23%** | **0.45** |

Metrics are arithmetic averages across seeds, not an ensemble. The full-feature model ranks better than both reduced-input models in all three seeds. SPY annualized at 11.64% on the same rebalance windows before benchmark costs; no variant's average cost-adjusted return exceeds it.

## Paired uncertainty

Use 4,000 circular block resamples within folds of the seed-averaged date-level IC differences. Ten-rebalance blocks are the primary interval; four-rebalance blocks are a sensitivity check. Intervals are conditional on fitted models, without multiple-comparison adjustment. Seeds share market dates and are not independent economic samples.

| Contrast | IC difference | 95% interval, 10-rebalance blocks | 95% interval, 4-rebalance blocks |
|---|---:|---|---|
| Macro state minus equity-only | -0.0052 | [-0.0277, +0.0165] | [-0.0322, +0.0229] |
| Full features minus macro state | +0.0272 | [+0.0031, +0.0520] | [-0.0022, +0.0563] |
| Full features minus equity-only | +0.0220 | [-0.0096, +0.0541] | [-0.0126, +0.0588] |

## Interpretation

The beta-feature bundle is promising as an input representation, even though supplying estimated beta magnitudes as graph-edge inputs did not help in the earlier controls. Macro-state features alone did not show an improvement here. Evidence for beta-input lift is not robust enough for a final positive claim: the full-versus-equity interval includes zero, and the full-versus-macro interval changes that conclusion with block length. The experiment does not separate signed betas, magnitudes and interactions from one another.

Next examine fold/regime stability and ablate the beta-derived components on these same development samples, with comparisons fixed before any further evaluation. Do not select a model from these results and label its development score final performance. Agree final confirmation dates with the team, including how to account for the standalone pilot's previous use of 2025.

Per-seed metrics, histories, predictions, input-group definitions, source manifests and configuration are retained locally; they are not included in this documentation update. This note reports aggregate findings only.
