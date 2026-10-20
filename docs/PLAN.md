# Plan

Pipeline status against the Infrastructure steps in `project_overview.md`.

## Done

- **Data stage (part of step 1):** `gnns build-data` fetches split-adjusted daily bars and
  dividend factors from Massive, adjusts, validates, and writes
  `data/<universe>/<bar>/prices.parquet` + `manifest.json`. Dry run on `sector_etfs` 1d,
  2010-01-01 to 2026-06-30: 42,030 rows, XLC/XLRE start at inception, XLK dividend
  adjustment implies ~1.3%/yr yield. Schema is DRAFT in `docs/CONTRACTS.md` until
  1m and 65m bars have been built. `data.universes` is a list built in one command;
  `benchmarks` (SPY, RSP) is its own universe.

- **Step 2, GraphSnapshot:** schema in `docs/CONTRACTS.md`, implemented in
  `src/gnns/data/snapshot.py` (validated on construction, lossless `.npz` round-trip,
  `add_reverse_edges` helper). FeatureEngine missing-value policy in
  `src/gnns/data/CLAUDE.md`.

- **Step 4, FeatureEngine (minimal, built before step 3):** `src/gnns/data/features.py`,
  features chosen per construction under `features:` in its graph file (no shared set). Momentum and volatility on the dense bar axis, `min_history_bars`
  eligibility, cross-sectional median fill, tested for look-ahead.

- **Point-in-time input rule** adopted for all builder inputs (`docs/CONTRACTS.md`);
  prices and features already comply.

- **Step 3, GraphBuilder + SectorETFCorrelationBuilder:** base class slices every input to
  its window ending at `t` (`src/gnns/graphs/base.py`); dummy builder is fully connected,
  weighted by |60-bar return correlation| with signed `corr` as an analysis-only edge
  column. Vectorized `pairwise_corr` in `src/gnns/graphs/stats.py` for reuse. Config:
  `graph:` selects `config/graphs/<name>.yaml`; one snapshot every `horizon_bars` (5).
  `gnns build-graphs` on sector_etfs: 818 snapshots (2010-03-31 to 2026-06-29), 9→11 nodes,
  72–110 edges, ~4 s.

- **Labels:** `src/gnns/data/labels.py`, `forward_returns(prices, horizon_bars)`; open of
  the next bar to open `horizon_bars` later, with `label_end` for purging.
  `horizon_bars: 5` in config. 41,960 labels on sector_etfs.

- **Step 5, WalkForwardSplitter:** `src/gnns/train/walk_forward.py`, config
  `walk_forward:`. Train | purge | val | purge | test, rolling or expanding. On sector_etfs:
  13 folds. Held-out period from `holdout_start: 2024-01-01` (shared config; folds
  anchored there): 10 search folds (tests 2013-12-26 to 2023-12-29), 3 held out.

- **Research protocol notes:** `docs/RESEARCH_PROTOCOL.md` separates grid-searched settings
  from the strategy-defining horizon (chosen by a horizon experiment) and experiment-design
  settings; records the
  fixed-recipe hyperparameter selection and the held-out period.

- **Env versions pinned:** direct dependencies in `environment.yml` pinned to exact
  versions (no lock file; conda still picks CUDA vs CPU torch per machine). Switch to
  `conda-lock` if indirect-package drift ever causes discrepancies.

- **Step 6, GraphSnapshotDataset:** `src/gnns/train/dataset.py`. One PyG `Data` per
  snapshot in a period: model columns selected, rank-normalized per node type, labels
  joined, unlabeled nodes kept and masked. Sector ETFs fold 1: 151 train / 26 val / 50 test
  samples; batched GATv2 forward runs on GPU.

- **Steps 7–8, models, Trainer, Evaluator, MLflow:** `gnns train`. Models `gatv2`,
  `graphsage`, `mlp` (`src/gnns/train/models.py`), loss `listmle` (`losses.py`), Adam with
  early stopping on validation rank IC (`trainer.py`), run orchestration + MLflow logging
  (`run.py`), metrics (`src/gnns/eval/`); charts on demand via `gnns report <run_id>`. Each construction trains its `model`
  and matched `baseline` on search folds only. First sector-ETF run (10 folds, ~2.3 min on
  GPU): mean test IC −0.004 (gatv2) vs −0.011 (mlp), i.e. no signal, as expected for the
  dummy graph; the pipeline itself works end to end.

## Next

- Collapsed scores / no learning on sector_etfs. Planted-signal tests
  (`tests/train/test_planted_signal.py`) show the training code learns. Found: with
  `lr: 0.001` + `patience: 10`, GATv2 cannot learn even a strong planted signal (test IC
  −0.06 vs 0.62): it starts flat and early stopping ends training before it moves.
  `lr: 0.01` or a much larger patience fixes it. Defaults now `lr: 0.01`, `patience: 20`,
  and `layers: 1` for the fully connected sector-ETF graph (2 layers oversmoothed).
  Next: rerun the sector-ETF baseline with these defaults.
- Finalize the data contract after 1m and 65m bars have been built.
- Step 9: real graph builders (correlation / graphical lasso first).
- Step 10: BacktestEngine (costs, turnover, drawdown, Sharpe/IR tearsheet).
- `gnns sweep` (grid search). When building it, first split `train/run.py:run()` along
  its seams: build (snapshots, labels, folds, datasets), train (each model per fold),
  summarize (metrics), log (MLflow). `gnns train` calls all four; the sweep builds once
  per graph/feature setting, trains each model combination on it, logs every run's
  metrics without predictions, and attaches predictions only for the best run by
  validation IC (keep-best-so-far in memory, no retraining).
- Ablation mechanism, built on `gnns sweep`: rerun a construction with a feature (or a
  correlated group, per the feature-correlation chart) removed, or with edges removed,
  across several seeds; compare mean validation IC paired by fold and seed. This is the
  decision-grade test for keeping or dropping inputs; permutation importance only shows
  how much a trained model relies on them.
- Extension: a richer interactive results viewer (beyond MLflow + PNGs), possibly a
  separate project.
- Universe open questions (CONTRACTS.md): ticker identity, survivorship, unadjusted close.
