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
| `horizon_bars`, `snapshot_every_bars` | define the strategy | separate experiments (below) |
| `train_bars`, `val_bars`, `test_bars` | experiment design | fixed up front; sensitivity checks only |
| held-out start | grading line | fixed up front; may move later as data grows, never earlier |

Only the first row is ever grid-searched. Never pick a setting from another row by its
test or held-out results.

### Alternative experiments (horizon and cadence)

Changing `horizon_bars` (e.g. 21 ≈ one month) or `snapshot_every_bars` (e.g. 1 = daily
rebalancing) defines a different strategy, not better settings for the same one. To try
them:

1. Pick candidate versions on economic grounds (the IC trades on weeks-to-months horizons).
2. Run each as a full experiment on the search folds, with its own tuning sweep inside.
3. Compare on cost-adjusted backtest metrics (Sharpe, IR, drawdown, turnover), not raw IC:
   ICs at different horizons are not comparable.
4. Report how many versions were tried; choose one (at most a few) before the held-out test.

When `horizon_bars > snapshot_every_bars`, consecutive labels overlap: the IC series is
autocorrelated and naive t-stats overstate significance; use overlap-adjusted errors.
These settings never switch between folds.

### Experiment-design sensitivity

Changing `train_bars`/`val_bars`/`test_bars` (e.g. 5-year training) checks whether a
conclusion is robust; it is not a way to pick the best-looking result.

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
