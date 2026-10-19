# gnns.graphs

The GraphBuilder abstract base class and all concrete builders. Every builder returns a
GraphSnapshot (`src/gnns/data/snapshot.py`). First implementation is a dummy sector-ETF
correlation builder that exercises the pipeline end to end.

## Decisions

- Builders implement `build_at(t, inputs)` and declare a lookback (in bars, or `None`) per
  input. The base class slices every input to its window ending at `t` (rule in
  `docs/CONTRACTS.md`, "Point-in-time inputs"), so a builder never sees rows after `t`.
- Builders never load or slice data themselves. A construction-specific source (commodity
  prices, BEA tables, filings, news) gets a loader that sets the known-at `timestamp`,
  with its own look-ahead test, and is then passed in like prices and features.
- Which timestamps are built is the run's choice, not the builder's: one snapshot every
  `horizon_bars` in `config/config.yaml`, shared by every construction.
- Each construction has `config/graphs/<name>.yaml` (builder, universe, `params`, its own
  node `features`, `sources`). Constructions share no features; config holds values,
  builder code declares structure (inputs, lookbacks). Shared across constructions:
  labels and snapshot spacing (`horizon_bars`), walk-forward and held-out settings.
- New builders are registered in `BUILDERS` in `build.py`; `params` become keyword
  arguments of the builder's `__init__`.
- `build_snapshots` checks the graph's `features.min_history_bars` covers the builder's
  price window.
- Use `stats.pairwise_corr` (vectorized, gap-aware) rather than per-pair loops.
- The sector-ETF builder is fully connected. Real builders on large universes must
  sparsify (graphical lasso, top-k): 1,500 fully connected nodes is ~2.25M edges/snapshot.

## Open questions

- Snapshots are built in memory at the start of each run; there is no disk cache. When a
  builder gets slow (e.g. graphical lasso on a large universe), consider caching
  snapshots to disk keyed by (builder config, dataset manifest sha256, code version).
  The snapshot schema is required to round-trip through a file losslessly, so a cache
  needs no schema change.
