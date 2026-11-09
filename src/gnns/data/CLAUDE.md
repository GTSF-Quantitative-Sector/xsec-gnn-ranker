# gnns.data

Universe (point-in-time tradeable set + entity resolution), FeatureEngine
(node-feature computation), and the GraphSnapshot schema (the locked interface every
graph builder must emit). This layer reads machine settings.

Every loader here outputs a `timestamp` column meaning when each row became known
(`docs/CONTRACTS.md`, "Point-in-time inputs"), and has a look-ahead test. Prices (bar end)
and features (computed at `t` from bars `<= t`) already comply.

Every dataset is a folder with one Parquet file and a `manifest.json` carrying the core
fields (`docs/CONTRACTS.md`, "manifest.json"). Every read goes through `store.open` (or
`store.load` for prices), which returns the manifest the caller logs. A source loader
calls `store.open`, then adds only its own checks. A source the pipeline builds has a
fetch function registered by `kind` in `build.SOURCES`: it fetches and validates, never
writes; `build()` writes every universe and source only after all of them validate.

## FeatureEngine decisions

- Computes equity node features for the whole price history at once, returning one long
  table `(timestamp, ticker, <feature columns>)`; builders filter it to their `t`.
  Features of construction-specific node types (commodities, macro, events) belong to
  the builder, not here.
- There is no shared feature set: each construction chooses its own in
  `config/graphs/<name>.yaml` under `features:` (`min_history_bars` + `list`). The feature
  functions are shared: a plain dict registry maps each `name` to one function, so a
  feature is computed identically wherever it is used. Columns are named from the entry
  (e.g. `momentum_20`).
- Windows count positions on the bar axis: each ticker is spread onto the full axis
  first, with missing bars as nulls.
- A ticker is eligible at `t` once it has at least `min_history_bars` bar-axis positions
  before `t` (per construction; must be at least the longest feature window).
- After that, a missing value is filled with that feature's cross-sectional median at
  `t`, and fill counts are logged. Add per-feature "was filled" indicator columns only if
  the logged counts turn out large.
- Outputs raw values. Cross-sectional normalization is the Dataset's job
  (`src/gnns/train/CLAUDE.md`).
