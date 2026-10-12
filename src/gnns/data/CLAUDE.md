# gnns.data

Universe (point-in-time tradeable set + entity resolution), FeatureEngine
(node-feature computation), and the GraphSnapshot schema (the locked interface every
graph builder must emit). This layer reads machine settings.

## FeatureEngine decisions

- A node enters snapshots only after `min_history_bars` bars (config; at least the longest
  feature lookback).
- After that, a missing value is filled with that feature's cross-sectional median across
  nodes of the same type at `t`. Each snapshot logs how many values were filled. Add
  per-feature "was filled" indicator columns only if the logged counts turn out large.
