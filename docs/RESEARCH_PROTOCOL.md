# Research protocol

## Point-in-time data

Nothing may use information that was not known at the time it is used.

- Every input carries a known-at `timestamp`; builders only receive rows known by `t`
  (`docs/CONTRACTS.md`, "Point-in-time inputs").
- Snapshots contain only what is known at `t`; labels (forward returns) live outside them.
- Every loader and feature has a look-ahead test: deleting all data after `t` must not
  change anything computed at or before `t`.

## Walk-forward evaluation

- Each fold is train → validation → test in time order, with a purge before validation and
  before test so no training label ends after the next period starts.
- Early stopping and hyperparameter choices use validation only, never test results.

## Settings: tuned vs fixed

`gnns sweep` runs two stages on the same seeds. Stage 1 tunes the GNN and the inputs both
models share; stage 2 tunes the baseline with those shared inputs fixed at the stage-1
winner, so it stays matched (same inputs, no graph). Each stage picks its winner by mean
validation IC over seeds. Tuned values are then frozen in the graph file (below).

| Where | Settings | Role | Stage 1 (GNN) | Stage 2 (baseline) |
| :-- | :-- | :-- | :-- | :-- |
| graph file §2 | `params.*`, `features.*`, `extra_model_features` | shared data side | **swept** | fixed at stage-1 winner |
| graph file §3 | `model.*`, `model.train.*`, `model_edge_attrs` | GNN | **swept** | not used |
| graph file §4 | `baseline.*`, `baseline.train.*` | baseline | not used | **swept** |
| graph file §2 | `features.min_history_bars` | derived (`>=` longest window) | set by the sweep | fixed at stage-1 winner |
| graph file §1 | `builder`, `universe`, `sources` | defines the construction | fixed | fixed |
| graph file §3–4 | `model.name`, `baseline.name` | architecture | fixed (separate sweeps) | fixed |
| config.yaml | `horizon_bars` | defines the strategy | fixed (horizon experiment below) | fixed |
| config.yaml | `walk_forward.*` (window, train/val/test bars) | experiment design | fixed (sensitivity checks only) | fixed |
| config.yaml | `walk_forward.holdout_start` | grading line | fixed (may move later, never earlier) | fixed |
| config.yaml | `data.*`, `normalize` | shared by every construction | fixed | fixed |
| config.yaml | `train.*` | training defaults; overridden per model | fixed | fixed |
| config.yaml | `train.seed` | noise, not a setting | `sweep.seeds`, averaged | same seeds |
| config.yaml | `eval.*`, `backtest.*` | evaluation only | fixed | fixed |
| derived | purge gap | `horizon_bars + 1` | fixed | fixed |

The graph file's sections: §1 construction, §2 shared data side, §3 GNN, §4 baseline,
§5 `sweep:`. Some builder params are controls or safety margins, not tuning choices, and
stay out of grids even though they sit in §2 (e.g. the macro graph's
`edge_mode: constant | shuffled`, `lag_bars`, `max_stale_bars`, `seed`); each graph doc
marks them.

Never pick a fixed setting by its test or held-out results.

`horizon_bars` is both the label horizon and the spacing between snapshots: one snapshot
(and one rebalance) every `horizon_bars` bars, so consecutive labels never overlap and each
rebalance's return is exactly one label.

### Horizon experiment

The horizon is chosen by experiment, not fixed in advance. For each candidate (e.g. 5, 10,
21, 63 bars):

1. Run the full pipeline with that `horizon_bars`, including its own grid search over the
   tuning settings, chosen on its search folds' validation as usual.
2. Compare the tuned candidates on the **search folds'** cost-adjusted backtest (Sharpe,
   information ratio, drawdown, turnover), with identical portfolio rules and costs. Not
   raw IC: ICs at different horizons are not comparable. Use annualized ICIR
   (`icir_annual`, logged per run) as the cross-horizon check on signal quality, and
   favor horizons that are consistent across years and survive doubled costs.
3. Choose the horizon (at most two finalists, named in advance) from step 2, then score the
   finalists once on the held-out period and report all of them, plus how many horizons
   were tried.

The held-out backtest confirms the choice; it does not make it. Choosing the horizon by its
held-out result would turn those years into another search period.

A horizon never changes between folds.

### Experiment-design sensitivity

Changing `train_bars`/`val_bars`/`test_bars` (e.g. 5-year training) checks whether a
conclusion is robust; it is not a way to pick the best-looking result.

## Baselines

Constructions share no node features: each chooses its own in its graph file, and some
may not use prices at all. What they share is the target (`forward_returns` over
`horizon_bars`, which also sets the snapshot spacing), the folds and held-out period, and
the Evaluator.

- **Matched baseline (required per construction):** the same node features and settings,
  with no graph (e.g. XGBoost or an MLP per node, or the same GNN with edges removed),
  configured in that construction's graph file. Construction minus matched baseline is
  the **graph lift**: the evidence that relationships carry signal.
- Cross-construction comparisons use graph lift, plus absolute results.
- **Price-only reference (optional):** a construction config using only price features
  (e.g. momentum, volatility), as a simple anchor for presenting results.

## Run-to-run noise

GPU training is not bit-for-bit reproducible (PyG's message passing sums in a varying
order), and a single seed is one draw of weight initialization and shuffling. Never
compare two settings on one run each: run every candidate with several seeds (e.g. 3,
via `train.seed`) and compare the averages, alongside their spread.

## Hyperparameter selection

Grid-searched settings are chosen once, from the search folds' average validation IC, and
frozen; model weights still retrain every fold. Prefer combinations that are good across
most folds and neighboring values. Check how often the best combination changes across
search folds: frequent changes mean differences are mostly noise, so pick a robust one.
`gnns sweep` logs, per stage, how many folds its winner won (`summary.yaml`) and every
combination's validation IC per fold and seed (`results.csv`) to check neighbors.

## Final held-out period

`walk_forward.holdout_start: 2024-01-01` in `config/config.yaml`, shared by every
construction and experiment so all are graded on the same years.

- Folds whose test period starts on or after it are held out (`Fold.holdout`). The splitter
  anchors fold layout so a test period starts exactly there, for any construction.
- Searches and tuning use search folds only. Held-out folds are scored once, for final
  results; looking earlier turns them into search years.
- The date is fixed in advance, never chosen from results. It may move later as data
  grows; never earlier.
- Sized from IC precision: the mean IC's standard error ≈ σ_IC / √(dates). On sector_etfs
  it gives ≈2.5 years (≈125 weekly dates, test periods 2024, 2025, 2026 so far),
  enough to detect a mean IC ≈ 0.05 across 11 names.
