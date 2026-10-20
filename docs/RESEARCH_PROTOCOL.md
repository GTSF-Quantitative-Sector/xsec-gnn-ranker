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

## Settings: tuned vs experiment-defining

| Setting | Role | How it is chosen |
| :-- | :-- | :-- |
| graph params (e.g. `window_bars`), feature windows, model params (lr, hidden size, heads, dropout) | tuning | grid search on search folds' validation, then frozen (below) |
| `min_history_bars`, purge gap | derived | follow from the settings above (`>=` longest window; `horizon_bars + 1`) |
| `horizon_bars` | defines the strategy | horizon experiment (below) |
| `train_bars`, `val_bars`, `test_bars` | experiment design | fixed up front; sensitivity checks only |
| held-out start | grading line | fixed up front; may move later as data grows, never earlier |

Only the first row is ever grid-searched. Never pick a setting from another row by its
test or held-out results.

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
