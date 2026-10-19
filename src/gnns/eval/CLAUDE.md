# gnns.eval

Evaluator (rank IC, quantile returns, top-k returns) and BacktestEngine (portfolio
weights, costs, tearsheet). Backtest is built last, once real model outputs exist.

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
  cascade, quantile equity curves, top-k equity curves (no costs), training curves.
- Every `gnns train` run logs to MLflow (`mlflow.db` + `mlruns/`, gitignored): resolved
  config, `git_sha` tag, dataset manifest, test IC metrics, `<model>_val_ic` (mean over
  folds of the kept epoch's validation IC: the metric to compare and select runs on;
  optimistic, so never reported as performance), training histories and predictions.
  No charts. `gnns train` always saves predictions; dropping them for losing sweep
  combinations is the future `gnns sweep`'s job (`docs/PLAN.md`).
- Charts are on demand: `gnns report <run_id>` (`report.py`) downloads that run's
  artifacts, calls `make_charts`, and logs the PNGs to the same run. It uses the run's
  own saved config, so charts match the settings the run was trained with.
- The predictions table is the hand-off to the BacktestEngine (step 10), which will read
  it the same way and never touch models or graphs.

## Open questions

- With fewer nodes than `eval.quantiles`, some quantiles are empty on those dates (e.g.
  Q6/Q11 before XLRE/XLC exist). Fine for a preview; revisit for the real universes.
- A richer interactive results viewer could be a separate project or extension later.
