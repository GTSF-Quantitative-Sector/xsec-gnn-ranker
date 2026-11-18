# Contracts

Locked interfaces between pipeline stages. Changing anything here means updating every
consumer. Status: **DRAFT** — finalize after the first dry run.

## Time and units

- Every length (window, horizon, rebalance interval, train/test fold, purge) is an
  integer count of **bars**, never a calendar unit. Config keys carry a `_bars` suffix,
  e.g. `window_bars: 60`.
- Bar size is a property of the dataset only, written `<int><unit>` with unit `m` or `d`
  (`1d`, `1m`, `65m`). Nothing downstream of the loader uses timedeltas or calendar math.
- Bars are counted on the dataset's **bar axis**: the sorted unique timestamps across all
  tickers in the file. Windows are positions on this axis, not per-ticker row counts.
- Timestamps mark the **end** of the bar, tz-aware `America/New_York`. Data stamped `t`
  is known at `t`; using rows with `timestamp <= t` is look-ahead-safe. Daily bars are
  stamped 16:00 ET, including early-close days (conservative).
- Annualization is the only calendar-aware step: `bars_per_year` is derived from bar size
  in one function (`1d` → 252, `65m` → 6 × 252), from the one year length
  `metrics.TRADING_DAYS`.

## Point-in-time inputs

Every table a graph builder reads (prices, features, and any construction-specific
source: commodity prices, BEA tables, filings, earnings events, news sentiment) has a
tz-aware `America/New_York` column **`timestamp` meaning when the row became known**.

- The loader that produces a source sets `timestamp` and owns its correctness, with its
  own look-ahead test. Examples: bar end for prices; release date for BEA tables; filing
  date (not period end) for filings; publication time for news.
- Sources that only give a date use a conservative documented rule (e.g. a filing dated
  `D` is known at `D` 16:00).
- Revised data is one row per version, each with its own `timestamp`.
- Data that never changes is stamped with the earliest date in the dataset.
- Builders never read a source directly. The GraphBuilder base class sorts each input by
  `timestamp` once, then passes `build_at(t, inputs)` only the rows each builder declared:
  for a lookback of `L` bars, rows with `axis[i - L] < timestamp <= t` (all rows
  `<= t` for `L = None`), where `axis[i] = t` on the universe's bar axis. For bar-aligned
  data that is exactly the `L` bars ending at `t`; irregular rows (news, filings) between
  bars fall into the next bar.
- Correct timestamps do not cover leaks inside a source's own construction, e.g. a
  sentiment model trained on later articles scoring earlier ones. Note these per source.

## Storage layout

**Every dataset** (prices, benchmarks, and any construction-specific source) is one folder
holding its data file (Parquet) and one `manifest.json` with the core fields ("manifest.json"
below). `data/` is gitignored.

Price datasets: one per (universe, bar size). Each universe is defined in
`config/universes/<universe>.yaml`, whose `kind` is one of:
- `list`: fixed `tickers`; every ticker is a node on every date it has a bar;
- `membership`: members change over time ("Membership universes" below).

**The central equity universe** is the membership universe `sp500`. Every construction's
tradeable nodes are this universe or a subset of it; other node types (macro factors,
commodities, …) come from construction-specific sources. Benchmarks live in their own
`benchmarks` list universe, built on the same bar size and range, and never appear in a
construction's universe. The `sector_etfs` list universe stays for the existing dummy
construction.

    data/<universe>/<bar>/
      prices.parquet
      manifest.json
    data/<universe>/membership/       # membership universes only
      membership.parquet
      manifest.json
    data/<universe>/delistings/       # membership universes only
      delistings.parquet
      manifest.json
    data/<universe>/reports/          # membership universes only: review reports (CSV)
    data/security_master/             # one for all membership universes
      security_master.parquet
      manifest.json
    data/wrds/<extract>/              # raw WRDS extracts the datasets above are built from
      <extract>.parquet
      manifest.json

WRDS extracts are written only by `gnns fetch-wrds`, which connects to WRDS (and may wait
for a Duo approval); `gnns build-data` only reads them. Extracts and everything built from
them stay under `data/` and are never committed. A membership universe's build also
writes review reports, one CSV per check that found rows (each build replaces them); they
fail nothing, and what they find is fixed in the reviewed override files below.

A construction-specific source is a folder under the data root, named in its graph file's
`sources:` (e.g. `data/macro/` with `macro.parquet` and `manifest.json`). A source whose
entry names a `kind` is built by `gnns build-data` while that graph is selected: the kind
selects a fetch function in `gnns.data.build.SOURCES`
(`fetch(spec, start, end) -> (data, manifest fields)`), which validates and returns the
dataset unwritten, and the build writes it as `<path>/<source name>.parquet` together with
the universes. Its manifest adds `bar`, `start` and `end` to the source's own fields.
Columns and fields per source are in that construction's doc (`docs/graphs/<name>.md`,
"Datasets needed").

A dataset is rebuilt in full, never appended to: new splits/dividends rescale history.
Writes are atomic (write temp, then rename), and nothing is written if validation fails.

## prices.parquet

Long format, one row per (timestamp, ticker), sorted by `timestamp` then `ticker`.

| Column | Arrow type | Notes |
| :-- | :-- | :-- |
| `timestamp` | `timestamp[us, America/New_York]` | bar end |
| `ticker` | `string` | as listed in the universe |
| `open`, `high`, `low`, `close` | `float64` | split- and dividend-adjusted |
| `volume` | `float64` | split-adjusted |

**Adjustment.** OHLC = Massive split-adjusted bars (`adjusted=true`) × the
`historical_adjustment_factor` of the first dividend whose `ex_dividend_date` is after the
bar's date (1.0 if none). Prices are adjusted as of `built_at`; levels differ between
builds but returns do not.

**Membership universes** adjust per `security_id`, because Massive adjusts per ticker and
never across a ticker change. OHLC = Massive's unadjusted daily bars, fetched for the
ticker valid on each date (security master, `ticker_massive`), × the product of the event
factors of every later event of the security. Volume is divided by the same product's
split part. Events, one factor each:

| Event | Source | Factor |
| :-- | :-- | :-- |
| split | Massive split records of the ticker valid on the execution date | `split_from / split_to` |
| cash dividend (regular or special) | Massive dividend records of the ticker valid on the ex-date | `1 − cash_amount / unadjusted close of the bar before the ex-date` |
| spinoff, up to CRSP's last date | CRSP distributions of type `SP` that don't end the security (not `SECMRG`, `SECEX`, `SECMWM`), through the `permno` link | CRSP's whole ex-date: `abs(price) / (abs(previous price) × (1 + total return))` |
| spinoff, after CRSP's last date | `config/universes/adjustment_overrides.csv` (reviewed, tracked) | the factor written there |

- On a date with a spinoff factor (CRSP or override), Massive's split and dividend records
  for that security and date are ignored: the spinoff factor already covers the whole day,
  and Massive records some spinoffs as splits (e.g. EBAY 2015-07-20, 1000:2376), which
  would count the event twice.
- Non-USD dividends, and dividends with no earlier bar or with cash not below that close,
  are skipped and reported. Bars that break the OHLC invariants are dropped and reported.
- `adjustment_overrides.csv` columns: `security_id, ex_date, factor, note`. A row is
  provisional: once CRSP's data covers its `ex_date`, the build uses CRSP's factor and
  reports the row as obsolete. Candidates come from the build's anomaly report: an
  overnight unadjusted move beyond a threshold on a day with no split or dividend, or a
  Massive split with a non-round ratio.
- The build reports, for every member-day CRSP covers, the share of days on which the
  adjusted close-to-close return matches CRSP's `dlyret` within a tolerance, and lists
  the security-days that don't (an aggregate check; it fails nothing).

**Missing bars** are absent rows, not NaN rows. Consumers that need a dense
bar-axis × ticker matrix pivot and get NaN where a ticker has no bar.

**Invariants** (checked before write):
- no null values
- `(timestamp, ticker)` is unique
- `0 < low <= min(open, close)` and `max(open, close) <= high`
- `volume >= 0`
- every `ticker` is in `manifest.tickers`; every `timestamp` is in `[start, end]`
- every ticker in `manifest.tickers` has at least one row

## Membership universes

A membership universe's file names its constituent list and how to turn it into nodes:

    # config/universes/sp500.yaml
    kind: membership
    source: datastream            # tr_ds_equities.ds2constmth (month-end lists)
    list_code: 4408               # S&P 500 (LS&PCOMP)
    one_class_per_company: true   # keep the most liquid share class of each company
    liquidity_window_bars: 126    # trailing window for "most liquid"

**Node key.** Its node key is `security_id`: the Datastream infocode, written as a decimal
string. It is stored in the `ticker` column of the universe's price dataset (and so of
labels and predictions), so every consumer works unchanged; in these datasets "ticker"
means "node key".

### membership.parquet

One row per membership interval, sorted by `security_id`, then `start`.

| Column | Arrow type | Notes |
| :-- | :-- | :-- |
| `security_id` | `string` | Datastream infocode |
| `start` | `timestamp[us, America/New_York]` | 16:00 on the first list date the security is on |
| `end` | `timestamp[us, America/New_York]`, nullable | 16:00 on the first later list date it is not on; null if it is on the latest list |

- A list dated `d` (the source's month-end, a trading day) is known at `d` 16:00. A
  security is a member at `t` when `start <= t < end` (or `end` is null). The source's
  own end dates are not used: on the latest list they are placeholders. List dates are
  the source spells' start and end dates; a month-end on which nothing changed is not one,
  and membership carries over it.
- A list entry the source gives no identity (no ticker and no CUSIP on any date, e.g.
  Datastream's "APARTMENT INV. & MAN. PLACEHOLDER") is not a security: it is dropped and
  reported.
- With `one_class_per_company`, each list keeps one security per company (Datastream
  company code): the one with the highest median of unadjusted close × volume over the
  `liquidity_window_bars` bars ending at `d`. The kept class may change between lists.
- A member that stops trading stays a member until its next list date; it simply has no
  more bars, and its delisting return reaches the labels (`delistings.parquet`).
- Invariants (the build fails, listing violations): `start < end`; one security's
  intervals don't overlap; with `one_class_per_company`, no company has two members on a
  list; every member session on which Datastream still gives the security a ticker has a
  `ticker_massive`, except holes of at most 5 sessions in a row (as around a seizure or a
  ticker change), which are reported. Every `security_id` is in the price dataset's
  `manifest.tickers`.

Manifest fields: `universe`, `list_code`, `one_class_per_company`, `first_list`,
`last_list` (ISO dates), `list_dates` (count).

### delistings.parquet

One row per security of the universe whose bars stop before the dataset's last session.

| Column | Arrow type | Notes |
| :-- | :-- | :-- |
| `ticker` | `string` | the security's node key (`security_id`) |
| `last_bar` | `timestamp[us, America/New_York]` | its last bar |
| `delret` | `float64` | return from the last bar's close to the final value |
| `source` | `string` | `crsp` or `last_trade` |
| `provisional` | `bool` | the last bar is after CRSP's last date |

- `source = crsp`: CRSP's delisting return of the PERMNO linked on the last bar's date
  (the nearest CRSP delisting from 10 days before to 60 days after it).
- `source = last_trade`: no such CRSP return, so the last traded price is the final value
  (`delret = 0`). Typically a security that kept trading OTC after its exchange delisting:
  its OTC bars are the realized path, past CRSP's record of the PERMNO. Rows up to CRSP's
  last date are listed in a review report.
- `provisional` rows are replaced by CRSP's return once CRSP's data covers them.

Manifest fields: `universe`, `crsp_last` (ISO date).

## Security master

One translation table for every security of every membership universe: dated links from
`security_id` (the primary key) to every other identifier. Code never joins sources on raw
tickers; it translates through this table on the date in question.

`security_master.parquet`, one row per (security, identifier type, validity window),
sorted by `security_id`, `id_type`, `valid_from`:

| Column | Arrow type | Notes |
| :-- | :-- | :-- |
| `security_id` | `string` | Datastream infocode |
| `id_type` | `string` | one of the types below |
| `value` | `string` | the identifier, in its source's own format |
| `valid_from`, `valid_to` | `date` | inclusive; `valid_to` null = still valid |
| `source` | `string` | `datastream`, `crsp`, `massive` or `override` |

| `id_type` | Source | Use |
| :-- | :-- | :-- |
| `cusip` | Datastream `ds2cusipchg` (8 characters) | the link to CRSP |
| `isin` | Datastream | reference |
| `company` | Datastream company code | one class per company |
| `permno`, `permco` | CRSP, via the `cusip` valid on the date | CRSP data (distributions, delisting returns, checks) |
| `ticker_crsp` | CRSP | reference |
| `ticker_datastream` | Datastream `ds2mnemchg` (dated ticker history; an end of 2079-06-05 means open) | source of `ticker_massive` |
| `ticker_massive` | `ticker_datastream` on the date, in the Massive spelling with the most bars in that ticker's interval (as written, or with `.` added or removed for a share class: `BRKB` → `BRK.B`). Within 7 days of a ticker change the neighbouring interval's symbol is a candidate too (Datastream sometimes switches a day early or late, and at a handover a symbol may still or already belong to another security): a neighbouring symbol is used only if its close is within a 20% log move of the security's previous close, and among candidates with a bar the nearest close wins, the interval's own symbol on ties within 0.02 | fetching Massive bars, splits and dividends |

- Each `id_type` is derived by one documented rule. Rows in the reviewed, tracked
  `config/universes/security_overrides.csv` (same columns plus `note`, `source` =
  `override`) replace derived rows for the same `security_id` and `id_type` on the dates
  they cover.
- Invariants (the build fails, listing violations): a security has at most one value per
  `id_type` on any date; a `cusip`, `isin`, `permno` or ticker value points to at most one
  security on any date (`company` and `permco` may be shared by several securities).
  Member sessions without a `permno` up to CRSP's last date are reported (they get no CRSP
  spinoff or delisting return).
- Identifier history is used only to locate data, never as a model input, so using a later
  rename to find earlier bars reveals nothing about returns.

Manifest fields: `securities` (count), `id_types`, `crsp_last` (ISO date).

## Loading

Every dataset read goes through `gnns.data.store`; nothing reads a dataset's Parquet
directly.

- `store.open(dest) -> (pl.LazyFrame, dict)` reads one dataset folder: its data file,
  lazily, and its manifest. It raises, naming the folder and "rebuild the dataset", if
  `manifest.json` is missing or lacks a core field. There is no hash check on read: the
  pipeline always writes the data file and its manifest together, and the logged
  `sha256` identifies the data version in each run.
- `store.load(root, universe, bar, start=None, end=None) -> pl.LazyFrame` opens one price
  dataset, optionally filtered to bar dates in `[start, end]`, and drops the manifest.
- `membership.mask(root, universe, axis) -> pl.DataFrame` returns `(timestamp, ticker)`
  for every bar-axis timestamp at which each security is a member; empty for list
  universes, which have no membership dataset.
- `security_master.lookup(sm, security_ids, id_type, on) -> pl.Series` translates security
  ids to another identifier on date `on` (null where none is valid);
  `security_master.resolve(sm, id_type, values, on) -> pl.Series` translates the other way;
  `security_master.attach(df, sm, id_type, key="ticker", date="timestamp")` adds the
  identifier valid on each row's date as a column named `id_type`.
- `store.write(df, dest, manifest, file="prices.parquet")` writes the data file and the
  manifest atomically, adding `file`, `rows` and `sha256`. It refuses a manifest without
  the other core fields.
- A construction-specific source's loader (in `gnns.data`) calls `store.open`, then adds
  only its own checks.

The engine is polars; consumers needing numpy or pandas convert at the edge.

## manifest.json

Core fields, in every dataset's manifest:

| Field | Type | Notes |
| :-- | :-- | :-- |
| `source` | str | where the data came from, e.g. `massive`, `fred` |
| `file` | str | the data file in the folder, e.g. `prices.parquet` |
| `rows` | int | row count of `file` |
| `sha256` | str | hash of `file` |
| `built_at` | str (ISO datetime, UTC) | when it was built; for prices, the adjustment reference date |
| `git_sha` | str | code version that built the dataset |

Each kind of dataset adds its own fields. Price datasets:

| Field | Type | Notes |
| :-- | :-- | :-- |
| `universe` | str | folder name |
| `bar` | str | e.g. `1d` |
| `start`, `end` | str (ISO date) | requested range, inclusive |
| `tickers` | list[str] | requested universe |

Every pipeline run logs to MLflow the manifest of each dataset it opened, as read when it
was opened (not re-read later), as `manifest_<name>.json`: `gnns train` logs `prices` and
each of the builder's sources; `gnns sweep` logs the same on every child run and on its
parent run; `gnns backtest` logs `backtest/manifest_benchmarks.json`.

## Labels

`forward_returns(prices, horizon_bars, delistings=None)` returns
`(timestamp, ticker, fwd_return, label_end)`, joined to snapshots on `(timestamp, ticker)`.
Training passes the universe's `delistings.parquet` when it has one
(`labels.load_delistings`).

- `fwd_return` at bar-axis position `i` is `open[i+1+h] / open[i+1] - 1`: enter at the next
  bar's open, hold `horizon_bars` bars. The backtest uses the same open-to-open convention.
- `label_end` is the timestamp of bar `i+1+h`, when the label is fully known; the
  WalkForwardSplitter purges on it.
- Only positions with `i + 1 + h <=` the last position have labels.
- **Delistings (#3):** a ticker whose exit bar is missing because its last bar `L` falls in
  the window (`i <= L < i+1+h`) is labeled from its final value `close[L] × (1 + delret)`:
  `fwd_return = close[L] × (1 + delret) / open[i+1] - 1`, or `delret` when `L = i`.
- Any other missing entry or exit bar (a halt, a data gap) gives no row; such nodes are
  left out of the loss and IC.
- **Predictions** (`predictions_<model>.parquet`) hold every ranked (`target_mask`) node at
  every date whose label window is complete, with a null `fwd_return` where the node has
  no label; dates in the last `horizon_bars + 1` bars give no rows. Every metric drops the
  null returns (`gnns.eval.metrics.labeled`); the backtest raises if it holds one.
- Raw returns. Excess-over-benchmark returns are a reporting choice in the Evaluator; they
  don't change rankings within a `t`.
- `horizon_bars` is shared config, identical for every construction.

## GraphSnapshot

The only object graph builders emit and the only graph input downstream stages consume.
One snapshot per bar-axis timestamp `t`. Arrays are numpy; the Dataset converts to torch
without casting.

**Rule: everything in a snapshot is known at `t`.** Labels (forward returns) are not part
of it — `gnns.data.labels.forward_returns` supplies them to training and eval.

### Fields

| Field | dtype, shape | Notes |
| :-- | :-- | :-- |
| `timestamp` | tz-aware datetime | bar end, `America/New_York` |
| `node_ids` | `str [N]` | unique; the price dataset's key for equities (ticker, or `security_id` in a membership universe), names for context nodes |
| `node_type` | `int64 [N]` | index into `node_type_names` |
| `node_type_names` | `tuple[str, ...]` | e.g. `("equity", "commodity")` |
| `target_mask` | `bool [N]` | nodes that are scored and ranked |
| `x` | `float32 [N, F]` | node features, column blocks per type (below) |
| `feature_names` | `str [F]` | unique |
| `feature_owner` | `int64 [F]` | node type that owns each column |
| `feature_prescaled` | `bool [F]` | `True` = the builder already scaled this column, so the Dataset leaves it as is; optional, defaults to all `False` |
| `edge_index` | `int64 [2, E]` | row 0 source, row 1 destination; messages flow source → destination |
| `edge_weight` | `float32 [E]` | 1.0 for unweighted edges |
| `edge_type` | `int64 [E]` | index into `edge_type_names` |
| `edge_type_names` | `tuple[str, ...]` | e.g. `("supplies", "exposes")` |
| `edge_attr` | `float32 [E, D]` | named edge columns; `D` may be 0 |
| `edge_attr_names` | `str [D]` | unique |

### Invariants (checked on construction)

- Equity nodes exist at `t`, pass the FeatureEngine's history requirement and, in a
  membership universe, are members at `t` (history counts every bar, including bars from
  before the security joined); a builder may drop more (e.g. the macro builder drops
  equities without enough data for their betas). Context nodes (e.g. macro factors) are
  never targets. All nodes are sorted by `(node_type, node_id)`. Row positions are not
  stable across snapshots; track nodes across time by `node_id`.
- **Column blocks:** `x[i, j] == 0` wherever `feature_owner[j] != node_type[i]`.
- `x`, `edge_weight`, `edge_attr` are finite; `edge_weight >= 0`.
- `edge_index` values are in `[0, N)`; no self-loops (models add their own).
- Undirected relations are stored in both directions with equal weight and attributes.
- A snapshot round-trips through a file losslessly. Files saved before `feature_prescaled`
  existed load with it all `False`.

### What reaches the model

`x` and `edge_attr` hold everything a builder computes, including columns meant only for
analysis (e.g. `beta_sign`, `hops_to_commodity`). The Dataset passes the model the
construction's `features` (its graph file's `features.list`) plus any builder-added node
columns listed in `extra_model_features`, and only the edge columns listed in
`model_edge_attrs`. Both lists default to empty, so analysis-only columns reach the model
only when listed explicitly.

The Dataset scales each model column across the nodes of its owning type at `t`, with the
shared `normalize` setting, except columns marked `feature_prescaled`, which pass through
unchanged. A builder marks a column prescaled when cross-sectional scaling would destroy it:
a value that is the same for every node at `t` (all ties under `rank`), or a column owned by
a node type with one node. It must have scaled the column itself using only data known by
`t`. Prescaled and raw columns can mix in one snapshot under any `normalize` setting.

## Open questions

- Whether to rename the `ticker` column to a neutral key name once every universe has
  one (today it holds a `security_id` in membership universes).
- Intraday: whether Massive's `65m` bars align to the 09:30 open; regular-hours filtering.
- Intraday: one missing bar empties a whole volatility window; thin names at `1m` may need
  partial windows (e.g. ≥ 80% of bars present) instead of frequent median fills.
