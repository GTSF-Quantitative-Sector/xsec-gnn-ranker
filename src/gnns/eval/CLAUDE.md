# gnns.eval

Evaluator (rank IC, quantile returns, top-k returns) and BacktestEngine (portfolio
weights, costs, tearsheet). Backtest decisions and steps: GitHub issue #2.

## Decisions

- Metrics (`metrics.py`) take a predictions table `(timestamp, ticker, score, fwd_return)`
  and never see models or graphs. IC is Spearman per date. Snapshots are `horizon_bars`
  apart, so labels never overlap: the plain `ic_t` is valid and equity curves can
  compound one label per rebalance.
- `ic_summary` reports `mean_ic`, `ic_t` (significance of the mean IC), `icir_annual`
  (mean / std of per-date IC times sqrt(rebalances per year), comparable across horizons),
  `hit_rate`, `n_dates`. `ic_t = icir_annual * sqrt(years of dates)`. Rebalances per year =
  `bars_per_year(bar) / horizon_bars`; `bars_per_year` is the one calendar-aware function.
- Charts (`plots.py`) are matplotlib PNGs, one table per model label so the model and its
  matched baseline share each chart: IC over time, cumulative IC, IC by fold, quantile
  cascade, quantile equity curves, training curves (portfolio charts: backtest below).
- Every `gnns train` run logs to MLflow (`mlflow.db` + `mlruns/`, gitignored): resolved
  config, `git_sha` tag, dataset manifest, test IC metrics, `<model>_val_ic` (mean over
  folds of the kept epoch's validation IC: the metric to compare and select runs on;
  optimistic, so never reported as performance), training histories and predictions.
  `gnns train` always saves predictions; dropping them for losing sweep
  combinations is the future `gnns sweep`'s job (`docs/PLAN.md`).
- Producers create a run (`train/run.py:run`, later `sweep`); consumers take a run id,
  read its artifacts and attach results to it (`report.py`, `backtest_run.py`). Producers
  never call consumers: `cli.py:_after_run` backtests and charts the run a command
  produced (`gnns train`: its run; future `gnns sweep`: its best run only, so test-period
  backtests never drive selection). `gnns report <id>` / `gnns backtest <id>` rerun one.
- `report.py` downloads the run's artifacts, calls `make_charts`, and logs the PNGs to
  the same run, using the run's own saved config so charts match its training settings.
- Permutation importance (`importance.py`) is computed inside `gnns train`, per fold on
  validation dates, while the trained model is in memory (weights are not saved).
  Scrambles stay within each date and node type; "edges" rewires who links to whom.
  Logged as `<model>_imp_<input>` and `importance.json`. Validation dates also chose the
  kept epoch, so like `val_ic` it is somewhat optimistic; it shows reliance, while the
  future ablation (`docs/PLAN.md`) decides what to keep.
- `gnns report` also computes the feature correlation chart from the price data (search
  period only, within-date rank correlation of the construction's `features.list`).
- The predictions table is the hand-off to the BacktestEngine, which never touches
  models or graphs.

## Backtest decisions

- Terms: a backtest "period" is one rebalance = one snapshot's holding window =
  `horizon_bars` bars = one label (snapshots are `horizon_bars` apart). Not a fold's
  train/val/test period. The per-year chart is grouped by rebalance date and shows each
  year's count (n): a fold starting in late December gives a one-rebalance stub year
  wherever the boundary is put, so it is shown, not hidden.
- `backtest.py` (engine, pure functions) and `backtest_run.py` (MLflow I/O, charts).
  Top-k equal weight, full rebalance each period; turnover = sum |target - drifted| (first
  rebalance buys from cash). Linear costs at every level in `cost_bps` (0 = gross);
  metrics keyed `<model>_<metric>_<bps>bps`. Gross return equals `metrics.top_k_returns`
  (tested), which is kept for that reconciliation.
- Benchmark (SPY) and risk-free (BIL, a T-bill ETF in `benchmarks`) returns come from
  `labels.forward_returns` over the run's horizon, so they cover the same open-to-open
  windows as the portfolio. Sharpe is over BIL, IR over SPY. Drawdown length is in
  trading days, comparable across horizons.
- The run's saved config gives `horizon_bars` and `data.bar`; the current `backtest:`
  block gives the portfolio rules, so old runs can be re-costed. Settings are logged as
  `backtest.*` tags and `backtest/backtest.yaml`; per-period tables
  (`backtest/periods_<model>.parquet`) allow hand checks.
- A held name with a null `fwd_return` raises (no fill, no skip, no config key). It
  cannot fire while predictions hold only labeled rows; trailing dates without labels
  must never reach predictions (issue #3).

## Open questions

- With fewer nodes than `eval.quantiles`, some quantiles are empty on those dates (e.g.
  Q6/Q11 before XLRE/XLC exist). Fine for a preview; revisit for the real universes.
- A richer interactive results viewer could be a separate project or extension later.
- Delisted names have no label, so they can never be selected: look-ahead in the
  backtest and IC on a real universe. Fix is upstream (issue #3).
- Shrinking `cost_bps` leaves the removed levels' metrics on a rerun run (MLflow cannot
  delete metrics); the `backtest.cost_bps` tag is the current list.
- Later options: rank hold buffer, rank-, volatility- or size-based weights (need a
  point-in-time input besides predictions), liquidity-aware costs, a yield-based
  risk-free series instead of BIL.
