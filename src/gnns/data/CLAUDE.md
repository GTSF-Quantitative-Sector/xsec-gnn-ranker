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

## Universe decisions (issue #1)

- Universes are `config/universes/<name>.yaml` with a `kind` (`universes.spec`): `list`
  (fixed tickers) or `membership`. `sp500` is the central equity universe; every
  construction's tradeable nodes are it or a subset of it.
- `wrds.py` is the only module that connects to WRDS (`gnns fetch-wrds`, Duo); it caches
  raw extracts under `<data_root>/wrds/`. Everything else reads the cache.
- `equities.py` builds a membership universe: `fetch_massive` (network) then `assemble`
  (pure, tested on fixtures). Steps: security master (`security_master.py`) -> one
  Massive symbol per session (`assign_symbols`: continuity decides ticker handovers) ->
  membership (`membership.py`: month-end lists, placeholders dropped, one class per
  company) -> event factors (Massive splits/dividends, CRSP spinoffs, overrides) ->
  adjusted prices -> delistings -> checks and review reports.
- `security_id` = Datastream infocode, stored in `ticker`. Code translates identifiers
  through the security master on the date in question, never by joining raw tickers.
- Fixes are data, never code: `config/universes/security_overrides.csv` and
  `adjustment_overrides.csv`, each row with a note.
- WRDS AI rules: Datastream (LSEG) rows may be inspected; CRSP data stays aggregate-only in
  an agent's context. Build output names securities by infocode and prints CRSP only as
  aggregates; the CRSP-derived review reports are for a person.

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
