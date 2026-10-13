# Contracts

Locked interfaces between pipeline stages. Changing anything here means updating every
consumer. Status: **DRAFT** — finalize after the first dry run.

## Time and units

- Every length (window, horizon, rebalance interval, train/test fold, purge) is an
  integer count of **bars**, never a calendar unit. Config keys carry a `_bars` suffix,
  e.g. `corr_window_bars: 400`.
- Bar size is a property of the dataset only, written `<int><unit>` with unit `m` or `d`
  (`1d`, `1m`, `65m`). Nothing downstream of the loader uses timedeltas or calendar math.
- Bars are counted on the dataset's **bar axis**: the sorted unique timestamps across all
  tickers in the file. Windows are positions on this axis, not per-ticker row counts.
- Timestamps mark the **end** of the bar, tz-aware `America/New_York`. Data stamped `t`
  is known at `t`; using rows with `timestamp <= t` is look-ahead-safe. Daily bars are
  stamped 16:00 ET, including early-close days (conservative).
- Annualization is the only calendar-aware step: `bars_per_year` is derived from bar size
  in one function (`1d` → 252, `65m` → 6 × 252).

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

One dataset per (universe, bar size). `data/` is gitignored. Each universe's ticker list
is tracked at `config/universes/<universe>.txt`, one ticker per line. Benchmarks live in
their own `benchmarks` universe, built on the same bar size and range as the construction
universes, and never appear in a construction's list (every universe ticker is a node).

    data/<universe>/<bar>/
      prices.parquet
      manifest.json

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

**Missing bars** are absent rows, not NaN rows. Consumers that need a dense
bar-axis × ticker matrix pivot and get NaN where a ticker has no bar.

**Invariants** (checked before write):
- no null values
- `(timestamp, ticker)` is unique
- `0 < low <= min(open, close)` and `max(open, close) <= high`
- `volume >= 0`
- every `ticker` is in `manifest.tickers`; every `timestamp` is in `[start, end]`
- every ticker in `manifest.tickers` has at least one row

## Loading

`gnns.data.store.load(root, universe, bar, start=None, end=None) -> pl.LazyFrame` scans one
dataset, optionally filtered to bar dates in `[start, end]`. The engine is polars;
consumers needing numpy or pandas convert at the edge.

## manifest.json

| Field | Type | Notes |
| :-- | :-- | :-- |
| `universe` | str | folder name |
| `bar` | str | e.g. `1d` |
| `start`, `end` | str (ISO date) | requested range, inclusive |
| `tickers` | list[str] | requested universe |
| `built_at` | str (ISO datetime, UTC) | adjustment reference date |
| `git_sha` | str | code version that built the dataset |
| `rows` | int | row count of `prices.parquet` |
| `sha256` | str | hash of `prices.parquet` |

Every pipeline run logs the manifest of each dataset it reads to MLflow.

## Labels

`forward_returns(prices, horizon_bars)` returns `(timestamp, ticker, fwd_return, label_end)`,
joined to snapshots on `(timestamp, ticker)`.

- `fwd_return` at bar-axis position `i` is `open[i+1+h] / open[i+1] - 1`: enter at the next
  bar's open, hold `horizon_bars` bars. The backtest uses the same open-to-open convention.
- `label_end` is the timestamp of bar `i+1+h`, when the label is fully known; the
  WalkForwardSplitter purges on it.
- No row where the entry or exit bar is missing; such nodes are left out of the loss and IC.
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
| `node_ids` | `str [N]` | unique; tickers for equities, names for context nodes |
| `node_type` | `int64 [N]` | index into `node_type_names` |
| `node_type_names` | `tuple[str, ...]` | e.g. `("equity", "commodity")` |
| `target_mask` | `bool [N]` | nodes that are scored and ranked |
| `x` | `float32 [N, F]` | node features, column blocks per type (below) |
| `feature_names` | `str [F]` | unique |
| `feature_owner` | `int64 [F]` | node type that owns each column |
| `edge_index` | `int64 [2, E]` | row 0 source, row 1 destination; messages flow source → destination |
| `edge_weight` | `float32 [E]` | 1.0 for unweighted edges |
| `edge_type` | `int64 [E]` | index into `edge_type_names` |
| `edge_type_names` | `tuple[str, ...]` | e.g. `("supplies", "exposes")` |
| `edge_attr` | `float32 [E, D]` | named edge columns; `D` may be 0 |
| `edge_attr_names` | `str [D]` | unique |

### Invariants (checked on construction)

- Nodes are exactly those that exist at `t` and pass the FeatureEngine's history
  requirement, sorted by `(node_type, node_id)`. Row positions are not stable across
  snapshots; track nodes across time by `node_id`.
- **Column blocks:** `x[i, j] == 0` wherever `feature_owner[j] != node_type[i]`.
- `x`, `edge_weight`, `edge_attr` are finite; `edge_weight >= 0`.
- `edge_index` values are in `[0, N)`; no self-loops (models add their own).
- Undirected relations are stored in both directions with equal weight and attributes.
- A snapshot round-trips through a file losslessly.

### What reaches the model

`x` and `edge_attr` hold everything a builder computes, including columns meant only for
analysis (e.g. `beta_sign`, `hops_to_commodity`). The Dataset passes only the columns
listed in config (`model_features`, `model_edge_attrs`); `model_edge_attrs` defaults to
empty, so edge attributes reach the model only when listed explicitly.

## Open questions

- Ticker identity across renames/reuse (plain ticker vs. a stable security id).
- Survivorship: fixed present-day ticker lists exclude names delisted before today.
- Unadjusted close column for price/liquidity universe filters.
- Intraday: whether Massive's `65m` bars align to the 09:30 open; regular-hours filtering.
- Intraday: one missing bar empties a whole volatility window; thin names at `1m` may need
  partial windows (e.g. ≥ 80% of bars present) instead of frequent median fills.
