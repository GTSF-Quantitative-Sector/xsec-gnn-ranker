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

## Next

- Finalize the data contract after 1m and 65m bars have been built.
- Step 3: GraphBuilder base class + SectorETFCorrelationBuilder.
- Universe open questions (CONTRACTS.md): ticker identity, survivorship, unadjusted close.
