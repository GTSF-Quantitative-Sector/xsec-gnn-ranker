# Contracts

Locked interfaces between pipeline stages. Changing anything here means updating every
consumer. Status: **DRAFT** — finalize after the first dry run.

## Time and units

- Every length (window, horizon, rebalance interval, train/test fold, purge, embargo) is an
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

## GraphSnapshot

TBD (infrastructure step 2).

## Open questions

- Ticker identity across renames/reuse (plain ticker vs. a stable security id).
- Survivorship: fixed present-day ticker lists exclude names delisted before today.
- Unadjusted close column for price/liquidity universe filters.
- Intraday: whether Massive's `65m` bars align to the 09:30 open; regular-hours filtering.
