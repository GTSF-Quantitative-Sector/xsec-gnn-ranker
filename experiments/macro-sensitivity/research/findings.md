# Macro-sensitivity: preliminary findings

## Research question

Do statistically estimated macro sensitivities add predictive information about sector-ETF returns? A later experiment will test whether a graph model uses this information more effectively than non-graph models. No GNN has been implemented or evaluated yet.

## Method

The prototype uses 11 sector ETFs and five factors: 10-year Treasury yield, high-yield OAS, VIX, Brent spot oil, and GLD adjusted prices as a gold proxy. ETF features include momentum and volatility. Joint rolling ridge regressions estimate signed macro betas over 252 sessions, requiring at least 200 complete observations. Factors are standardized within each trailing regression window. Macro features and betas use an assumed five-session delay.

Predictions are formed after close t for a five-session return from close t+1 through close t+6. All three ridge baselines use the same eligible assets and dates, ticker indicators, and training-only preprocessing. Split boundaries purge overlapping forward labels. Settings are recorded in ../config.json.

## Validation results

Recorded from the September 22, 2026 prototype run. There were 232 usable training dates (July 18, 2024–June 20, 2025) and 122 validation dates (July 1–December 22, 2025).

| Model | Mean Rank IC | Mean gross five-session top-three minus bottom-three spread |
|---|---:|---:|
| Equity features | 0.048063 | 0.183331% |
| Equity + macro features | 0.017586 | 0.161483% |
| Equity + macro + betas and interactions | 0.025410 | 0.272807% |

Betas and interactions improve the gross spread relative to both simpler models, but do not beat the equity-only model's Rank IC. This is mixed preliminary evidence, with no statistical-significance claim. The final holdout remains unscored.

## Limitations

- The available credit series restricts the common history; this is not a multi-regime study.
- Latest-vintage macro data and assumed release delays do not establish point-in-time validity.
- GLD is a proxy for gold; the study uses ETFs rather than individual company equities.
- The linear macro-only addition is shared across assets on a date and cannot directly change their rankings; nonlinear comparisons are still needed.
- Forward-return spreads overlap. They are not daily portfolio returns and exclude transaction costs, turnover, and financing costs.
- Confidence intervals, multiple development folds, shuffled-beta controls, and graph comparisons are not implemented.
- Fresh downloads may differ due to revisions and the moving credit-history coverage window. Locally cached source manifests record retrieval time, coverage, and hashes; raw data is not distributed here.

## Validation and next steps

Six automated tests cover future-data invariance of historical features, next-close execution labels, split-boundary purging, recovery of known signed exposures, pre-inception missing data, and basis-point conversion.

Next: resolve longer credit history and historical availability, add nonlinear and shuffled-beta baselines, evaluate across chronological development folds, quantify uncertainty and trading costs, then test GATv2. Keep the final holdout out of model selection.
