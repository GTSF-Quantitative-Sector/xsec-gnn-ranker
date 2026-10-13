# Graph development protocol

## Scope

This is the sector-ETF pilot of the macro-sensitivity graph. It tests whether estimated exposures help and whether graph/temporal processing improves on nonlinear non-graph baselines. It does not establish independence from disclosed firm exposures, compare against an implemented supply-chain graph, or reproduce the original THGNN study.

## Graph and architecture

Each snapshot has 11 ETF targets and five context-only macro sources, with 55 directed macro-to-ETF edges. Each factor has a distinct learned encoder for its level and change. ETFs have a shared feature encoder and learned ticker embeddings. Macro nodes are never scored; no macro labels or equity-to-equity edges are used.

PyTorch Geometric's GATv2Conv performs multi-head bipartite attention using the rolling beta as an edge attribute. The edge attribute influences attention logits; it is not a fixed multiplicative message weight. An ETF residual path preserves its own features. Bipartite self-loops are disabled. The snapshot model uses the last date only, even though its eligibility dates match the sequence models.

The temporal adaptation runs that graph encoder on each of 20 historical snapshots, then a shared unidirectional GRU processes each ETF's graph representations in date order. A scalar prediction head scores each ETF at the sequence endpoint. Factor-specific encoders provide relation distinctions. This is a purpose-built GATv2+GRU adaptation, not the authors' THGNN code or a paper-faithful benchmark.

References: [GATv2 operator and edge attributes](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html), [THGNN paper](https://arxiv.org/abs/2305.08740), [authors' THGNN repository](https://github.com/TongjiFinLab/THGNN). No code from the THGNN repository is vendored.

## Model matrix

| Name | Inputs / question |
|---|---|
| equity_mlp | ETF features and ticker identity |
| macro_mlp | Adds macro levels/changes |
| beta_mlp_magnitude | Adds absolute betas and macro-change interactions |
| beta_mlp_signed | Signed-beta nonlinear baseline |
| gat_magnitude | Snapshot GATv2; magnitude-only primary graph |
| gat_signed | Signed edge attributes |
| gat_constant | All edge attributes equal; learned attention remains active |
| gat_shuffled | Entire exposure vectors shuffled across ETFs independently by date |
| temporal_magnitude | GATv2+GRU using historical magnitudes |
| temporal_signed | Temporal signed-edge variant |
| temporal_constant | Temporal graph with uninformative edge attributes |
| temporal_shuffled | Temporal graph with shuffled exposure assignments |
| temporal_frozen | Each sequence repeats its first available beta snapshot throughout; nodes still evolve |
| sequence_mlp | Non-graph beta/macro MLP followed by GRU on the same 20 days |

Shuffles are seeded and computed once on the date grid, so overlapping sequences see the same altered graph on a shared date. They destroy both asset assignment and persistent temporal exposure structure. The constant and frozen controls address different questions. Magnitude-only models receive no beta signs; signs in attention exports are original, unshuffled beta metadata only, not the model's transformed edge input.

## Data and timing

Raw data is truncated to December 31, 2025 before graph features or targets are constructed. Thus, the 2026 holdout is neither fitted nor scored. Complete 11-ETF snapshots and 20 consecutive valid equity sessions are required. Missing feature dates invalidate the sequence; the calendar is never compressed to skip gaps. These stricter rules differ from the original ridge pilot's nine-asset minimum, so its old results are not a matched comparator.

Beta construction, five-session assumed publication lag, and next-close execution match the original pipeline. Historical release availability and vintages are still unverified. This remains exploratory latest-vintage research.

The first development fold trains through June 2025 and evaluates Q3; the second expands training through September and evaluates Q4. Labels extending across a boundary are removed. Later evaluation sequences may use earlier evaluation-period features, as would be available at prediction time, but weights and preprocessing stay fixed within each fold. Previously purged labels may become available for a subsequent fold after they mature.

Scalers use unique historical feature dates in the training sequences only. Edges use a positive training RMS scale per factor, preserving signs or nonnegative magnitudes. All models have identical target dates. Labels for optimization are cross-sectionally demeaned forward returns, divided by a training-only standard deviation. Reported spread metrics use the original forward returns.

## Training and uncertainty

The fixed budget is 30 epochs, hidden size 16, two attention heads, AdamW, and three seeds (7, 19, 43). There is no validation-based early stopping or hyperparameter selection in this runner. Parameter counts are saved; equal epochs do not imply exactly matched capacity. CPU deterministic operations and seeds are enabled. Exact library versions and code/data hashes are saved in run.json.

Rank IC is daily cross-sectional Spearman correlation. The spread is the average forward return of the top three minus bottom three ETFs. Each seed's metrics are computed separately; the summary averages those metrics rather than evaluating an ensemble of predictions.

Intervals use 1,000 moving-block resamples with 10-session blocks within each fold, after averaging seeds by date. Paired intervals resample date-matched IC differences. They are conditional on the fitted models, based on a short sample, and not corrected for multiple comparisons or past experimentation. They should not be interpreted as confirmatory evidence.

## Regimes and attention

High/low VIX uses the training-endpoint median of the lagged VIX input. Rising versus flat/falling rates uses the sign of the lagged yield level's 21-session change, not an ex-post policy-cycle label. Attention exports preserve date, ETF, factor, model, fold, and seed. By-regime summaries average those weights.

For temporal models, exported attention is from the last graph snapshot only. It does not measure the importance of earlier snapshots to the GRU, and attention weights are not causal factor attribution. Factor-removal retraining is a future robustness check.

## Reproducibility and next gate

The full runner creates checkpoints, preprocessing JSON, predictions, daily metrics, attention, regime tables, training histories, and plots. These stay out of Git. Store selected aggregate findings under research/ for review. A fresh source download can change the common sample because credit history is a moving window; matching the original cached manifest is necessary to replicate historical results exactly.

Before a final holdout evaluation: obtain longer credit history, verify point-in-time timing, freeze the model/hyperparameters, and add a portfolio simulation with turnover/costs. The original ridge script's explicit holdout option remains separate; the graph runner intentionally has none.
