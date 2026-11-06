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

## Run and sweep decisions

- `run.py` is split along its seams: `build(cfg) -> Built` (snapshots, labels, search
  folds, per-fold datasets, manifests), `train(built, role, cfg, device) -> Trained` (one
  model, `model` or `baseline`, on every fold; keeps each fold's best-epoch model on CPU),
  `importance`, `summarize(results, cfg, test=)` and `log_run(..., predictions=,
  parent_run_id=)`. `run()` (`gnns train`) calls them for both models.
- Training settings per model = config.yaml `train:` with the graph file's
  `<role>.train:` on top (`train_settings`). `seed` is shared and rejected there;
  `build_model` ignores the `train` key.
- `sweep.py` (`gnns sweep`, issue #9): stage 1 trains only the GNN on every `model_grid`
  combination x seed, building once per distinct data-side (§2) setting; stage 2 trains
  only the baseline on every `baseline_grid` combination x seed on the stage-1 winner's
  `Built` (no rebuild). Winner = highest mean `val_ic` over seeds, ties to the earlier
  combination in grid order (`better`). Allowed keys per grid: the table in
  `docs/RESEARCH_PROTOCOL.md`, "Settings: tuned vs fixed" (`check_key`).
- Keep-best-so-far holds the winner's `Trained` per seed (predictions and models), so the
  final runs (one per seed, both winners) need no retraining; permutation importance is
  computed only there, from the kept models. Stage runs log validation metrics only.
- `features.min_history_bars` is set for every combination to the largest any
  `model_grid` combination needs (`required_history`, never below the file value), so all
  are scored on the same folds and node sets. `required_history` reads a builder's
  `lookbacks()` by constructing it on empty inputs, so builders must not touch data in
  `__init__` (both current builders only store and validate params).
- Only the main process writes MLflow; jobs run one after another (parallelism: #17).

## Dataset decisions

- `dataset.build_dataset(snapshots, labels, start, end, cfg) -> list[Data]` for one
  period (e.g. a fold's train range). One sample = one snapshot (a whole cross-section at
  one `t`), as a PyG `Data` with `x`, `edge_index`, `edge_weight`, `edge_attr`, `y`,
  `label_mask`, plus `node_ids` and `timestamp`. `label_mask` = ranked (`target_mask`) and
  has a label. Batch with `torch_geometric.loader.DataLoader`; graphs stay disjoint.
- Model node columns = the construction's `features.list` columns plus builder-added
  columns listed in its graph file's `extra_model_features` (default `[]`); edge columns
  only those in `model_edge_attrs` (default `[]`).
- `y` is the raw forward return (`gnns.data.labels`); losses that need ranks rank inside
  the loss.
- Nodes without a label stay in the graph (they still pass messages) and are excluded
  from the loss and IC via `label_mask`: whether a label exists depends on bars after `t`.
- Normalizes those features cross-sectionally at each `t`, per node type, using one
  shared function (`dataset.normalize`, so analysis can reproduce what the model saw).
  Config `normalize: rank | zscore | none` in `config/config.yaml`, set to `rank` (percentile scaled to [-1, 1], ties
  averaged, a lone node gets 0); `zscore` is clipped at ±3.
- Columns the builder marked `feature_prescaled` skip that scaling and reach the model as
  the builder computed them; the rest of the snapshot is still scaled with `normalize`.
  The model and the baseline see the same values, because both go through `to_data`.
- Single-node types and values shared by every node at `t` can't be normalized
  cross-sectionally (rank makes them 0). The builder scales them itself, as time-series
  scaling from data known by `t`, and marks them `feature_prescaled`.
