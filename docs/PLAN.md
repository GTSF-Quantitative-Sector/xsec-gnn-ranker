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
  `graph:` selects `config/graphs/<name>.yaml`; `snapshot_every_bars: 5`.
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
  from strategy-defining (horizon, cadence) and experiment-design settings; records the
  fixed-recipe hyperparameter selection and the held-out period.

- **Env versions pinned:** direct dependencies in `environment.yml` pinned to exact
  versions (no lock file; conda still picks CUDA vs CPU torch per machine). Switch to
  `conda-lock` if indirect-package drift ever causes discrepancies.

- **Step 6, GraphSnapshotDataset:** `src/gnns/train/dataset.py`. One PyG `Data` per
  snapshot in a period: model columns selected, rank-normalized per node type, labels
  joined, unlabeled nodes kept and masked. Sector ETFs fold 1: 151 train / 26 val / 50 test
  samples; batched GATv2 forward runs on GPU.

## Next

- Finalize the data contract after 1m and 65m bars have been built.
- Step 7: ModelFactory + LossFactory registries (GATv2, GraphSAGE; ListMLE) and the
  Trainer (early stopping on validation).
- Universe open questions (CONTRACTS.md): ticker identity, survivorship, unadjusted close.
