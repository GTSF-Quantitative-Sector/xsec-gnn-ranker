# gnns.train

WalkForwardSplitter (date-based folds with label purging), GraphSnapshotDataset
(PyG-compatible batching), ModelFactory and LossFactory registries, and the Trainer
loop. This layer reads machine settings.

## WalkForwardSplitter decisions

- `walk_forward(axis, cfg["walk_forward"], cfg["horizon_bars"]) -> list[Fold]`; a `Fold`
  is six inclusive timestamps (train, val, test start/end) plus `holdout`. Bar positions
  only; it never sees snapshots or labels. Pass the bar axis from the first snapshot onward.
- Layout: train | gap | val | gap | test, gap = `horizon_bars + 1`, so every
  label ends before the next period starts. Test periods tile in steps of `test_bars`,
  anchored so one starts exactly at the required `holdout_start`; folds testing from it on
  are `holdout=True`. The last may be partial. `rolling` or `expanding` training window;
  bars before the first fold's training window are unused in both.
- Training and tuning code must skip `holdout=True` folds unless a run is explicitly the
  final held-out evaluation.
- Validation is for early stopping and tuning only (`docs/RESEARCH_PROTOCOL.md`).

## Dataset decisions

- Passes the model only the configured `model_features` / `model_edge_attrs` columns.
- Normalizes those features cross-sectionally at each `t`, per node type, using one
  shared function (so analysis can reproduce what the model saw). Config
  `normalize: rank | zscore | none`, default `rank` (percentile scaled to [-1, 1], ties
  averaged, a lone node gets 0); `zscore` is clipped at ±3.
- Single-node types can't be normalized cross-sectionally; they'll need per-fold
  time-series scaling once the supply-chain and macro builders exist.
