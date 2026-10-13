# gnns.data

Universe (point-in-time tradeable set + entity resolution), FeatureEngine
(node-feature computation), and the GraphSnapshot schema (the locked interface every
graph builder must emit). This layer reads machine settings.

Every loader here outputs a `timestamp` column meaning when each row became known
(`docs/CONTRACTS.md`, "Point-in-time inputs"), and has a look-ahead test. Prices (bar end)
and features (computed at `t` from bars `<= t`) already comply.

## FeatureEngine decisions

- Computes the shared equity features for the whole price history at once, returning one
  long table `(timestamp, ticker, <feature columns>)`; builders filter it to their `t`.
  Features of construction-specific node types (commodities, macro, events) belong to
  the builder, not here.
- Features are a config list of `{name, <params>}` entries; `name` maps to a function in
  a plain dict registry, and columns are named from the entry (e.g. `momentum_20`).
- Windows count positions on the bar axis: each ticker is spread onto the full axis
  first, with missing bars as nulls.
- A ticker is eligible at `t` once it has at least `min_history_bars` bar-axis positions
  before `t` (config; must be at least the longest feature window).
- After that, a missing value is filled with that feature's cross-sectional median at
  `t`, and fill counts are logged. Add per-feature "was filled" indicator columns only if
  the logged counts turn out large.
- Outputs raw values. Cross-sectional normalization is the Dataset's job
  (`src/gnns/train/CLAUDE.md`).
