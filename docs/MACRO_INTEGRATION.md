# Macro-sensitivity integration

This adds a snapshot macro construction to the existing pipeline. The original standalone
pilot is retained separately. No real-data training or final holdout evaluation is part of
this implementation. The temporal GATv2+GRU pilot is not yet migrated.

## Reused pipeline

The construction reads shared OHLCV prices, uses FeatureEngine equity features, returns
GraphSnapshot objects, and is selected through the existing builder registry. Dataset,
next-open labels, horizon-based snapshot spacing, walk-forward splitter, GATv2, MLP,
ListMLE, trainer, evaluator and MLflow reporting remain shared. The tracked default
construction and fold configuration are unchanged. Local construction selection is now
resolved after machine overrides, rather than replacing an already resolved graph with
a string.

## Required macro source

Place `known_at.csv` and `known_at.json` under `<data_root>/macro/`, or change `sources.macro.path`
in the macro graph configuration. CSV columns are exactly:

```csv
timestamp,factor,value
2010-01-04T17:00:00-05:00,rates,3.85
```

The example is illustrative, not a claim about a real release time. Factors are `rates`,
`credit`, `vix`, `oil`, `gold`. Each row is an event updating that factor's latest known
state; timestamp is its actual availability time with an explicit UTC offset. An event
after close cannot affect the preceding close's graph. Events between bars enter at the
next bar. Macro values are carried forward for at most `max_stale_bars` universe bars.
No initial values are backward filled. Duplicate timestamp/factor rows are rejected.

The sidecar must include:

```json
{
  "source": "provider and exact series identifiers",
  "availability_method": "document how historical known-at states were established",
  "known_at_verified": true,
  "units": {"rates": "percent", "credit": "percent", "vix": "index", "oil": "price", "gold": "price"},
  "sha256": "SHA-256 of the exact CSV bytes"
}
```

`known_at_verified` is the upstream producer's assertion; the loader cannot independently
prove publication timing from a Boolean. Record evidence in the sidecar, including URLs,
retrieval/version dates, timezone rules, price proxy definitions and licensing details.
The entire source sidecar is logged with the run. With revised sources, reconstruct the
latest known state upstream: an old observation revised today must not replace today's
current observation as if it were new. The current event schema does not itself implement
a vintage database or a vendor downloader. Do not relabel the legacy date/value cache as
verified merely by adding a five-session offset or setting this Boolean.

## Graph and matched information

Eligible equity nodes have the configured equity feature history and enough complete
return/macro observations for the joint ridge fit. Five context-only macro nodes send
directed messages to equities. Context identity is explicit, independent of node order.
All nodes are sorted by `(node_type, node_id)`; only equities are targets.

Returns and macro changes use the shared bar axis. Rates/credit changes are basis points,
VIX changes index points, and oil/gold changes fractional returns. Ridge uses trailing
factor-standardized changes and centered equity returns. `lag_bars` adds a further lag
after recorded source availability; it does not infer release timing. A complete window
of bar positions is required, with `min_obs` complete regression observations. Missing
equity prices are not filled; eligibility is per asset.

Macro levels are centered/scaled on trailing available history, macro changes divided by
trailing change standard deviations, and betas expressed per trailing one-SD factor move.
These transforms use only history available at the snapshot's lagged endpoint. The
construction requires `normalize: none`, preventing cross-sectional rank normalization
from destroying factor identity or macro levels. Equity features retain their shared raw
FeatureEngine units in this first integration.

Both GATv2 and MLP receive equity-owned columns containing macro state, signed betas,
magnitudes and beta/change interactions. Context-only columns are zero on equities.
Consequently the MLP sees the economic information available through graph context,
even though it ignores edges. This tests graph lift beyond those supplied inputs, not
beta signal versus a price-only model. Extra model columns are explicitly selected in
the graph file.

Nonnegative edge_weight carries absolute beta; the named edge attribute `beta` carries
the configured signed or magnitude value. GATv2 uses those as attention attributes,
not fixed multiplicative message weights. `constant` replaces both channels with ones;
`shuffled` permutes whole equity exposure vectors with a deterministic date-local seed.
Node exposure features stay unchanged in these topology controls. Factor-specific edge
types are metadata: the generic GATv2 does not dispatch relation-specific parameters.
It also retains the shared model's self-loops. This differs from the pilot's factor-specific
encoders and bipartite residual architecture, and is a new experiment specification.

## Running and acceptance

Select `graph: macro_sensitivity` and `normalize: none` in local `config/machine.yaml`.
Use normal shared commands documented in README. A bad source, incompatible normalization
or no eligible snapshots produces an explicit error. If snapshots begin too late, the
existing splitter rejects the absence of a complete search fold. Do not shorten folds
or move the holdout boundary merely to make the short credit cache train.

The cached pilot credit series begins September 2023 and yields no usable pre-2024
features. Obtain longer approved history and shared price/benchmark datasets before a
real run. The prior pilot already evaluated 2025, inside the team's 2024-onward holdout;
disclose that overlap when agreeing final confirmation dates.

Tests use synthetic dates, small fold/window settings and generated prices/macros. They
check timing, missingness, layout, labels, matched feature access, edge controls,
serialization, disjoint batches, finite gradients, shared training, source provenance
logging, reports and exclusion of held-out prediction dates. No performance inference
can be drawn from fixture metrics.
