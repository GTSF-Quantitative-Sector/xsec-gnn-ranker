# macro_sensitivity

Config: [`config/graphs/macro_sensitivity.yaml`](../../config/graphs/macro_sensitivity.yaml).
Builder: `macro_sensitivity` ([`src/gnns/graphs/macro.py`](../../src/gnns/graphs/macro.py));
macro dataset build and loader: [`src/gnns/data/macro.py`](../../src/gnns/data/macro.py).
Select it locally with `graph: macro_sensitivity` in `config/machine.yaml`, then use the
shared commands in the README; `gnns build-data` then also builds the macro dataset.
Shared labels, horizon, folds, holdout and `normalize` settings are unchanged.

## Hypothesis

Statistically estimated macro sensitivities (rolling ridge betas of each ETF on rates,
credit, volatility, oil and gold) add predictive information about cross-sectional
returns, and a graph model connecting macro context nodes to equities uses that
information better than a non-graph model given the same inputs. This tests graph lift
beyond the supplied macro inputs, not beta signal versus a price-only model.

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
- **Regression:** returns and macro changes use the shared bar axis. Every factor is a
  price series ("Data decisions" below), so every macro change is its fractional return;
  `invert_factors` (rates) use minus that return. Ridge uses trailing factor-standardized
  changes and centered equity returns. `lag_bars` adds a further lag after the bar-end
  availability; it does not infer release timing. A complete window of bar positions is
  required, with `min_obs` complete regression observations. Missing equity prices are
  not filled; eligibility is per asset.
- **Equity features:** the graph file's `features.list`, plus equity-owned columns
  containing macro state (`macro_level_*` for `level_factors` only, `macro_change_*` for
  all five), signed betas (`beta_*`), magnitudes (`magnitude_*`) and beta/change
  interactions (`interaction_*`). Both GATv2 and the MLP baseline receive them
  (`extra_model_features`), so the MLP sees the economic information available through
  graph context, even though it ignores edges.
- **Macro node features:** `context_level` (0 on factors outside `level_factors`),
  `context_change` and a one-hot `factor_*` identity; zero on equities.
- **Scaling:** macro levels are centered/scaled on trailing available history, macro
  changes divided by trailing change standard deviations, and betas expressed per trailing
  one-SD factor move, using only history available at the snapshot's lagged endpoint. The
  builder marks the columns it scaled itself `feature_prescaled` (macro level, change,
  interaction, context and factor identity), so the shared `normalize` leaves them as is
  and still ranks equity features, betas and magnitudes across equities.

## Data decisions

The pilot used FRED series and a Yahoo GLD download, with an assumed five-session release
lag. This construction uses Massive series on the team's current plan instead, so every
input comes from one vendor, is a closing price known at the 16:00 ET bar end (never
revised, no assumed lag), and exists at any bar size, including the minute bars planned in
#7. The pilot's results are therefore not directly comparable. Decided in #16; pre-study
numbers are 2009-2026 daily data.

| Factor | Pilot | Here | Why |
| :-- | :-- | :-- | :-- |
| rates | FRED `DGS10` (10y yield) | IEF (7-10y Treasuries), inverted | IEF's daily return tracks the 10y yield change (correlation 0.96, duration ≈ 7.4y). Minus its return keeps "yields up" positive, as with the yield. Massive's own daily 10y yield is the same number as `DGS10`, but has a date and no time (an assumed availability time) and no intraday version. |
| credit | FRED `BAMLH0A0HYM2` (ICE high-yield spread) | HYG (high-yield corporates) | FRED only serves about 3 years of the ICE spread (licensing), which left no pre-2024 training history; HYG starts 2007. HYG alone, not a ratio to Treasuries: its rate sensitivity is about zero (slope on IEI 0.01, sign flipping by regime), so HYG/IEI added rate exposure (correlation with IEF −0.41 vs −0.04). The joint ridge already holds the rates factor fixed. |
| vix | FRED `VIXCLS` | VXX (short-term VIX futures), joined across its 2019 note change | Spot VIX needs a paid Massive Indices tier. Daily VXX returns correlate 0.89 with VIX % changes (0.93 on days VIX moves over 10%); the builder's betas agree closely (`beta_vix` correlation 0.90, same sign 99.9%, median per-date rank agreement across ETFs 0.95). VXX moves about half as much as VIX; per-window scaling removes that. |
| oil | FRED `DCOILBRENTEU` (Brent spot) | BNO (Brent futures) | Massive has no spot crude. BNO matches the pilot's Brent; USO (WTI) lost 63% vs BNO's 33% in spring 2020 when front-month WTI went negative. BNO starts 2010-06-02. |
| gold | GLD (Yahoo) | GLD (Massive) | Same series. |

- **Levels dropped for rates, credit and vix** (`level_factors: [oil, gold]`). VXX loses
  value steadily to the futures roll (its trailing z-scored price is below zero on 88% of
  days, vs 64% for VIX), and IEF and HYG are total-return prices that trend. Their price
  levels say nothing about whether yields, spreads or volatility are high, so none of
  their levels reaches the model (no synthetic stand-in, which would look like a level
  without being one). Their changes, betas and interactions are kept. Oil and gold keep
  their price levels, as in the pilot.
- **The VXX join** (`sources.macro.series.vix.stitch`): the original VXX note matured on
  2019-01-29 and its replacement traded as VXXB until it took the VXX name on 2019-05-02,
  leaving 64 trading days with no VXX bar. Unfilled, that gap removed every beta from
  2019-04-17 to 2020-02-10. The series uses old VXX through 2019-01-29, VXXB through
  2019-05-01 (liquid by then, 0.93 correlation with VIX), and new VXX after. Each switch
  is joined on returns: VXXB's own return from its 2019-01-29 bar; for the rename, the
  return between VXXB's and VXX's unadjusted closes (+0.15%). VXXB is not used earlier:
  in 2018 it traded only 10-200 times a day and its closes were stale. `gnns build-data`
  refuses a switch-day return above 25%.
- Alternatives worth testing on daily data, recorded under "Future experiments" in
  [docs/RESEARCH_PROTOCOL.md](../RESEARCH_PROTOCOL.md): Cboe's official daily VIX history
  instead of VXX (also gives a true volatility level), and Massive's 10y yield instead of
  IEF.

## Datasets needed

- `sector_etfs` prices and `benchmarks` (`gnns build-data`).
- **Macro dataset** `<data_root>/macro/` (`sources.macro.path`): built by `gnns build-data`
  while this graph is selected (`sources.macro.kind: macro`), from the tickers in
  `sources.macro.series`, over the same `data.start`/`data.end` as prices. Each ticker's
  daily bars are split- and dividend-adjusted like prices; the build fails without writing
  anything if a series is missing a factor, has a non-positive or non-finite value, or a
  join is broken (see `gnns.data.macro.fetch_macro`).

  `macro.parquet` has exactly `timestamp` (tz-aware, the 16:00 ET bar end), `factor`,
  `value`. Factors are `rates`, `credit`, `vix`, `oil`, `gold`; `value` is the adjusted
  close (for vix, the joined series in the first VXX note's price units). Each row is an
  event updating that factor's latest known state; `timestamp` is its availability time.
  Events between bars enter at the next bar. Macro values are carried forward for at most
  `max_stale_bars` universe bars. No initial values are backward filled. Duplicate
  timestamp/factor rows are rejected.

  Besides the core fields, the manifest records `source` (the tickers),
  `availability_method`, `known_at_verified: true`, `units` (every factor `price`),
  `series` (the config used, including the join dates), `coverage` (first and last date per
  factor), `bar`, `start` and `end`. `load_macro` refuses a dataset whose manifest lacks
  these or has other units: rebuild it with `gnns build-data`. The whole manifest is logged
  with each run (`manifest_macro.json`).

## Params

| Param | Meaning |
| :-- | :-- |
| `window_bars` | returns per ridge regression window |
| `min_obs` | complete observations required in the window |
| `ridge` | ridge penalty |
| `lag_bars` | additional bar lag after bar-end availability |
| `max_stale_bars` | bars a macro value is carried forward |
| `edge_mode` | `magnitude`, `signed`, `constant` or `shuffled` |
| `seed` | seed for `shuffled` |
| `invert_factors` | factors whose change is minus their return (`[rates]`); part of the factor definition, not tuned |
| `level_factors` | factors that get `macro_level_*` and a nonzero `context_level` (`[oil, gold]`); not tuned |

## Limitations

- BNO starts 2010-06-02, so the first snapshot with a complete 252-bar window falls in
  mid-2011 rather than early 2011.
- VXX understates the size of the most extreme volatility days (futures move less than
  spot VIX in a spike), and as an exchange-traded note its price can diverge from the
  futures it tracks when the issuer limits creation.
- The prior pilot already evaluated 2025, inside the team's 2024-onward holdout; disclose
  that overlap when agreeing final confirmation dates.
- If snapshots begin too late, the splitter rejects the absence of a complete search fold.
  Do not shorten folds or move the holdout boundary to make a run fit.

## Status

Integrated into the shared pipeline, with the macro dataset built by `gnns build-data`; no
real-data training or final holdout evaluation yet. The temporal GATv2+GRU model is not
yet migrated. Tests (`tests/graphs/test_macro.py`) use synthetic dates, small fold/window
settings and generated prices/macros. They check timing, missingness, layout, labels,
matched feature access, edge controls, level and inversion settings, serialization,
disjoint batches, finite gradients, shared training, source provenance logging, reports
and exclusion of held-out prediction dates. `tests/data/test_macro_build.py` and
`tests/data/test_build.py` check the dataset build against a fake Massive client (joins,
dividend adjustment, refused joins, look-ahead, all-or-nothing writes). No performance
inference can be drawn from fixture metrics.

Earlier standalone pilot: [write-ups](../research/macro-sensitivity-pilot/), code at tag
`macro-sensitivity-pilot`.
