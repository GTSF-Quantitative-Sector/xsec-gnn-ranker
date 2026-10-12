# gnns.graphs

The GraphBuilder abstract base class and all concrete builders. Every builder's
`build(date, universe)` must return a GraphSnapshot (see `src/gnns/data/`). First
implementation is a dummy sector-ETF correlation builder that exercises the pipeline
end to end.

## Open questions

- Snapshots are built in memory at the start of each run; there is no disk cache. When a
  builder gets slow (e.g. graphical lasso on a large universe), consider caching
  snapshots to disk keyed by (builder config, dataset manifest sha256, code version).
  The snapshot schema is required to round-trip through a file losslessly, so a cache
  needs no schema change.
