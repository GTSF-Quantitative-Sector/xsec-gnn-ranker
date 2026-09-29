# Macro-Sensitivity Graph

First milestone: reproducible data ingestion, rolling macro exposures, and three non-graph ridge baselines. The GATv2 model is a later experiment.

## Run

Requires Python 3.11+ with the packages in requirements.txt.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python macrograph.py download
python macrograph.py run
```

Run these commands from the experiment directory. In the shared repository, first run `cd experiments/macro-sensitivity`.

See [preliminary findings](research/findings.md) for the initial validation results and limitations.

Default output is outputs/validation/REPORT.md. Raw downloads are cached in data/raw with retrieval timestamps, URLs, coverage, and SHA-256 hashes. Delete the relevant cached CSV and JSON to refresh a source; changing the configured date range also refreshes it. Data and generated outputs are excluded from Git.

## Experiment contract

- Universe: 11 sector ETFs, with no price filling before inception. Dates need at least nine eligible assets; all models use the same complete-case sample.
- Requested history: January 2010 through September 21, 2026. This does not imply all factors cover that period.
- Macro nodes: DGS10, BAMLH0A0HYM2, VIXCLS, DCOILBRENTEU, plus GLD as an explicit gold-price proxy. GLD is not spot gold. Curve slope is deferred.
- Credit: FRED currently supplies only three years of this ICE OAS series. The common sample is consequently short. There is no automatic substitution for credit.
- Changes: rates and OAS in basis points, VIX in index points, Brent and GLD in fractional returns. Levels are retained separately. At most two missing source-grid rows are forward-filled; equity prices are never filled.
- Beta: joint trailing 252-session ridge regression with at least 200 complete observations; penalty 10. Each factor is standardized inside its historical regression window. Signed coefficients represent return sensitivity to a one-window-SD move, rather than raw physical-unit betas. Magnitudes are separately available for the planned graph.
- Timing: asset features through close t; macro features and estimated betas lagged five equity sessions. Signal after close t, assumed execution at close t+1, target through close t+6. Missing release timestamps mean this is an assumed delay, not proven point-in-time availability.
- Baselines: equity features; equity plus macro features; equity plus macro plus signed betas and beta/change interactions. All use ticker indicators and a fixed ridge penalty of 100. Preprocessing is fitted only on training rows.
- Training: through June 30, 2025, purging labels extending beyond that boundary. Validation: July–December 2025, likewise purged. Later data is an unscored holdout by default. Hyperparameters have not been optimized.
- Metrics: mean daily cross-sectional Spearman Rank IC and mean top-three minus bottom-three forward return. Spreads overlap; they are not additive daily strategy returns. No inference, cost-adjusted returns, or turnover estimates yet.

## What the first results can establish

The prototype checks whether the code and data are usable and produces preliminary validation comparisons. Latest-vintage macro values, retrospectively adjusted prices, a short credit history, and assumed release lags prevent a production-quality historical claim. A longer approved OAS history with release/vintage information is needed for multi-regime research.

The equity-plus-macro baseline is intentionally transparent: a shared linear macro term is constant across assets within a date and cannot directly change their ranks. Correlated covariates can still change fitted equity coefficients. Add a nonlinear baseline before comparing against a GNN.

## Next milestones

1. Resolve longer credit history and verify release timing; audit gaps and source definitions.
2. Inspect beta stability; compare 126/252/504-session windows using development data only.
3. Add nonlinear tabular baselines, factor-removal controls, block-based uncertainty, and an explicit turnover/cost portfolio simulation.
4. Add a macro-to-equity GATv2 with signed versus magnitude edge attributes and constant/shuffled edge controls.
5. Freeze the protocol before explicitly running `python macrograph.py run --holdout`. Never use holdout results for model selection.
6. Consider temporal architecture only after the snapshot comparison works.

## Sources

- [FRED OAS coverage restriction](https://fred.stlouisfed.org/series/BAMLH0A0HYM2)
- [Treasury yield](https://fred.stlouisfed.org/series/DGS10)
- [VIX](https://fred.stlouisfed.org/series/VIXCLS)
- [Brent spot](https://fred.stlouisfed.org/series/DCOILBRENTEU)
- [ALFRED vintage data](https://alfred.stlouisfed.org/help/downloaddata)

Yahoo Finance's public chart endpoint supplies adjusted ETF closes; it is a prototype adapter and may require maintenance. No account credentials or paid services are used. Source data remains local.
