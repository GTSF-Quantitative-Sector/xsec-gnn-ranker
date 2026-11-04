# macro_sensitivity

Config: [`config/graphs/macro_sensitivity.yaml`](../../config/graphs/macro_sensitivity.yaml).
Builder: `macro_sensitivity` ([`src/gnns/graphs/macro.py`](../../src/gnns/graphs/macro.py));
loader: [`src/gnns/data/macro.py`](../../src/gnns/data/macro.py). Select it locally with
`graph: macro_sensitivity` in `config/machine.yaml`, then use the shared commands in the
README. Shared labels, horizon, folds, holdout and `normalize` settings are unchanged.

## Hypothesis

Statistically estimated macro sensitivities (rolling ridge betas of each ETF on rates,
credit, VIX, oil and gold) add predictive information about cross-sectional returns, and
a graph model connecting macro context nodes to equities uses that information better
than a non-graph model given the same inputs. This tests graph lift beyond the supplied
macro inputs, not beta signal versus a price-only model.

## Nodes, edges, features

- **Nodes:** eligible equities (targets) plus five context-only macro nodes
  (`macro:<factor>`), never scored. Eligible equities have the configured equity feature
  history and enough complete return/macro observations for the joint ridge fit. Context
  identity is explicit, independent of node order. All nodes are sorted by
  `(node_type, node_id)`.
- **Edges:** directed macro → equity, one per factor per equity, edge type = factor.
  Nonnegative `edge_weight` carries absolute beta; the edge attribute `beta` carries the
  configured signed or magnitude value. GATv2 uses those as attention attributes, not
  fixed multiplicative message weights. `edge_mode: constant` replaces both channels with
  ones; `shuffled` permutes whole equity exposure vectors with a deterministic date-local
  seed. Node exposure features stay unchanged in these topology controls. Factor-specific
  edge types are metadata: the generic GATv2 does not dispatch relation-specific
  parameters, and it retains the shared model's self-loops. This differs from the pilot's
  factor-specific encoders and bipartite residual architecture, and is a new experiment
  specification.
- **Regression:** returns and macro changes use the shared bar axis. Rates/credit changes
  are basis points, VIX changes index points, and oil/gold changes fractional returns.
  Ridge uses trailing factor-standardized changes and centered equity returns. `lag_bars`
  adds a further lag after recorded source availability; it does not infer release timing.
  A complete window of bar positions is required, with `min_obs` complete regression
  observations. Missing equity prices are not filled; eligibility is per asset.
- **Equity features:** the graph file's `features.list`, plus equity-owned columns
  containing macro state (`macro_level_*`, `macro_change_*`), signed betas (`beta_*`),
  magnitudes (`magnitude_*`) and beta/change interactions (`interaction_*`). Both GATv2
  and the MLP baseline receive them (`extra_model_features`), so the MLP sees the economic
  information available through graph context, even though it ignores edges.
- **Macro node features:** `context_level`, `context_change` and a one-hot `factor_*`
  identity; zero on equities.
- **Scaling:** macro levels are centered/scaled on trailing available history, macro
  changes divided by trailing change standard deviations, and betas expressed per trailing
  one-SD factor move, using only history available at the snapshot's lagged endpoint. The
  builder marks the columns it scaled itself `feature_prescaled` (macro level, change,
  interaction, context and factor identity), so the shared `normalize` leaves them as is
  and still ranks equity features, betas and magnitudes across equities.

## Datasets needed

- `sector_etfs` prices and `benchmarks` (`gnns build-data`).
- **Macro source:** a dataset folder `<data_root>/macro/` (or change `sources.macro.path`)
  holding `macro.parquet` and `manifest.json` (`docs/CONTRACTS.md`, "Storage layout"). It
  is produced outside the pipeline, so the loader also checks the file's SHA-256 against
  the manifest. Columns are exactly `timestamp` (tz-aware), `factor`, `value`. Factors are
  `rates`, `credit`, `vix`, `oil`, `gold`. Each row is an event updating that factor's
  latest known state; `timestamp` is its actual availability time. An event after close
  cannot affect the preceding close's graph. Events between bars enter at the next bar.
  Macro values are carried forward for at most `max_stale_bars` universe bars. No initial
  values are backward filled. Duplicate timestamp/factor rows are rejected.

  Besides the core fields, the manifest must include:

  ```json
  {
    "source": "provider and exact series identifiers",
    "availability_method": "document how historical known-at states were established",
    "known_at_verified": true,
    "units": {"rates": "percent", "credit": "percent", "vix": "index", "oil": "price", "gold": "price"}
  }
  ```

  Write the folder with `gnns.data.store.write(events, data_root / "macro", manifest,
  file="macro.parquet")`, which adds `file`, `rows` and `sha256` (the manifest also needs
  `built_at` and `git_sha`). `known_at_verified` is the upstream producer's assertion; the
  loader cannot independently prove publication timing from a Boolean. Record evidence in
  the manifest, including URLs, retrieval/version dates, timezone rules, price proxy
  definitions and licensing details. The whole manifest is logged with each run
  (`manifest_macro.json`). With revised sources, reconstruct the latest known state
  upstream: an old observation revised today must not replace today's current observation
  as if it were new. There is no vintage database or vendor downloader yet. Do not relabel
  the legacy date/value cache as verified merely by adding a five-session offset or
  setting this Boolean.

## Params

| Param | Meaning |
| :-- | :-- |
| `window_bars` | returns per ridge regression window |
| `min_obs` | complete observations required in the window |
| `ridge` | ridge penalty |
| `lag_bars` | additional bar lag after verified availability |
| `max_stale_bars` | bars a macro value is carried forward |
| `edge_mode` | `magnitude`, `signed`, `constant` or `shuffled` |
| `seed` | seed for `shuffled` |

## Limitations

- The cached pilot credit series begins September 2023 and yields no usable pre-2024
  features. Obtain longer approved history before a real run. The prior pilot already
  evaluated 2025, inside the team's 2024-onward holdout; disclose that overlap when
  agreeing final confirmation dates.
- If snapshots begin too late, the splitter rejects the absence of a complete search fold.
  Do not shorten folds or move the holdout boundary merely to make the short credit cache
  train.

## Status

Integrated into the shared pipeline; no real-data training or final holdout evaluation
yet. The temporal GATv2+GRU model is not yet migrated. Tests
(`tests/graphs/test_macro.py`) use synthetic dates, small fold/window settings and
generated prices/macros. They check timing, missingness, layout, labels, matched feature
access, edge controls, serialization, disjoint batches, finite gradients, shared
training, source provenance logging, reports and exclusion of held-out prediction dates.
No performance inference can be drawn from fixture metrics.

Earlier standalone pilot: [write-ups](../research/macro-sensitivity-pilot/), code at tag
`macro-sensitivity-pilot`.
