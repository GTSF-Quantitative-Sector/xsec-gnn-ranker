# sector_etf_corr

Config: [`config/graphs/sector_etf_corr.yaml`](../../config/graphs/sector_etf_corr.yaml).
Builder: `sector_etf_correlation` ([`src/gnns/graphs/sector_etf.py`](../../src/gnns/graphs/sector_etf.py)).

Every construction has one doc like this, with these sections.

## Hypothesis

None to test. This is the dummy construction (`docs/project_overview.md`, Infrastructure
step 3). It runs the whole pipeline end to end on a small universe before BEA tables,
segment data or macro feeds exist, and its results are not evidence for or against
relational signal.

## Nodes, edges, features

- **Nodes:** the 11 GICS sector ETFs (`config/universes/sector_etfs.txt`), one node type
  `equity`. Every node is a target. A node enters once it has `features.min_history_bars`
  of history, so XLRE and XLC join later than the rest.
- **Edges:** fully connected. One undirected edge per pair, stored in both directions,
  type `correlated`. `edge_weight` is the absolute Pearson correlation of daily close-to-close
  returns over the last `window_bars` returns ending at `t`. The signed correlation is the
  edge column `corr`. A pair is skipped if it has fewer than 3 shared returns or no variance.
- **Node features:** the graph file's `features.list` (momentum 20 and 60, volatility 20).
  The builder adds no columns and marks none prescaled.

## Datasets needed

- `sector_etfs` prices (`gnns build-data`).
- `benchmarks` prices for the backtest (SPY, BIL).
- No extra `sources:`.

## Params

| Param | Meaning |
| :-- | :-- |
| `window_bars` | returns per correlation window. Must be `<= features.min_history_bars`, which `build_snapshots` checks. |

## Limitations

- 11 nodes. Cross-sectional IC on 11 names is noisy, and the held-out period is sized
  for a mean IC of about 0.05 (`docs/RESEARCH_PROTOCOL.md`).
- A fully connected graph with one GAT layer already reaches every node in one hop, so
  the graph adds little structure beyond a correlation-weighted average.
- Correlation edges are mostly market beta. There is no sparsification or partial
  correlation.

## Status

Pipeline exercise. Used to build and check the data, training, evaluation and backtest
stages. It is not a research candidate.
