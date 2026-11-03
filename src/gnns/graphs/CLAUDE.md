# gnns.graphs

The GraphBuilder abstract base class and all concrete builders. Every builder returns a
GraphSnapshot (`src/gnns/data/snapshot.py`). First implementation is a dummy sector-ETF
correlation builder that exercises the pipeline end to end.

## Decisions

- Builders implement `build_at(t, inputs)` and declare a lookback (in bars, or `None`) per
  input. The base class slices every input to its window ending at `t` (rule in
  `docs/CONTRACTS.md`, "Point-in-time inputs"), so a builder never sees rows after `t`.
- Builders never slice data themselves. A construction-specific source (commodity
  prices, BEA tables, filings, news) gets a loader in `gnns.data` that calls `store.open`
  and sets the known-at `timestamp`, with its own look-ahead test, and is then passed in
  like prices and features.
- **Builder hooks** (classmethods on `GraphBuilder`, no-ops by default), so shared code
  never names a builder (no `if graph["builder"] == ...` in `build.py`):
  - `check_config(cfg)` raises if the resolved config can't work with the construction.
    `build_snapshots` calls it right after the unknown-builder check, before any data is
    loaded, so it sees config values only.
  - `load_sources(root, sources) -> {name: (frame, manifest)}` loads the graph file's
    `sources:` by calling the `gnns.data` loaders. Each frame becomes an input named
    `name` (`prices` and `features` are reserved). Each manifest is returned by
    `build_snapshots` and logged by the run as `manifest_<name>.json`.
- A builder marks node columns it already scaled with `feature_prescaled` (one flag per
  column; `docs/CONTRACTS.md`, "What reaches the model"), e.g. a macro value copied onto
  every stock, which ranking would erase. It does that scaling itself from data known by
  `t`; every other column goes through the shared `normalize`.
- `build_snapshots` drops snapshots with no target node, logging the count and the first
  and last dropped date, and raises if none remain. Return an empty snapshot only during
  warm-up or when the builder's inputs are genuinely unavailable at `t`: a dropped date
  leaves a gap in the `horizon_bars` grid, which the backtest skips rather than counting
  as cash.
- Which timestamps are built is the run's choice, not the builder's: one snapshot every
  `horizon_bars` in `config/config.yaml`, shared by every construction.
- Each construction has `config/graphs/<name>.yaml` (builder, universe, `params`, its own
  node `features`, `sources`) and `docs/graphs/<name>.md` (template:
  `docs/graphs/sector_etf_corr.md`), plus a row in the README's constructions table. Constructions share no features; config holds values,
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
