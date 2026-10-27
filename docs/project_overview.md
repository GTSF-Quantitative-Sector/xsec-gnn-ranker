# GTSF IC Quant Sector: GNN Project Plan

GitHub: https://github.com/GTSF-Quantitative-Sector/xsec-gnn-ranker

## Goal

Standard cross-sectional factor models treat each stock as an independent observation, discarding the economic and statistical relationships between assets that we observed to be significant in prior work on centrality and correlation-driven strategies. This project tests whether explicitly modeling those relationships through a graph neural network (GNN), conditioned on each stock's predicted rank on both its own features and the behavior of its graph neighbors, produces a cross-sectional ranking signal that meaningfully outperforms a standard non-relational factor model on a long-only equities universe. Rather than committing to a single definition of "relationship" upfront or a single definition of our set of nodes, we will construct and evaluate several distinct graph representations in parallel and empirically determine which relational structures, if any, carry real predictive signal. Success is defined by several factors including a validated ranking model, a decile backtest showing a genuine cascading return structure from top to bottom, and a simulated long-only portfolio whose risk-adjusted performance (Sharpe, information ratio, drawdown) is strong enough (when adjusting for transaction costs, which are fairly negligible on a tax-free portfolio with moderately infrequent trades) to justify a recommendation to the Investment Committee for allocating real endowment capital to the automated trading strategy.

## Graph Construction Ideas

Rather than committing to one definition of "relationship" between assets, we will build and evaluate the following graph constructions in parallel, using the shared model/target/evaluation framework described in the "Methods" tab on the left. Each idea specifies its own node universe, edge definition, and node/edge features, but is evaluated on identical terms (at the very least, as there may be further additions depending on the graph's specific context) so results remain comparable across ideas.

### General Roadmap:

For the first couple of weeks (done by September 14), the team will focus on learning basics regarding GNNs, exploring papers/other sources to brainstorm different directions we can take with the project (in terms of what values to assign nodes/edges, what sets of nodes/edges to use, specific GNN models like GAT vs GraphSAGE, etc.), and build a consistent infrastructure for the data pipeline, environments, and evaluation engine (based on the metrics we listed above) to ensure that our results between different team members are equally accurate and reproducible. We will likely have to create a basic correlation GNN (such as by simply connecting the different sector ETFs) to have a concrete example to build the evaluation engine around. While the brainstorming will be ongoing throughout the semester, it will be a main focus during these early weeks.

We will then divide the top 3 (based on team member interest/vote) GNN constructions among the team members to work on (ideally in pairs) in parallel, with frequent testing being done using the shared evaluation engine. This process will contain a lot of iteration, feedback from senior analysts / sector heads, and further brainstorming/ideation to see where the theme can be taken. In terms of deliverables, I would like to see each iterative cycle to be complete in about 2 weeks, with some milestone/construction progress ready to show each week, but if a certain iteration seems more difficult or time consuming, then please communicate a more realistic timeline with the team. This process will end either once there is a conclusive result supporting the GNN construction OR there is a conclusive result showing that the GNN construction has no predictive power. More ideas can be picked up for research & development using the same general iterative process outlined above.

## Methods:

- **Data:**
  We will use Massive API as the data source and each team member will store their relevant data in parquet files on their local device that are gitignore'd (layout in docs/CONTRACTS.md). As for the time increments, daily data from [01-2010, 07-2026) will most likely yield the most reliable results for the IC's long-only equities trading horizon (that is normally of the order of weeks to months), and we can further experiment later in the semester with tighter time chunks (on the order of hourly or even minutely) depending on the context/ideas for the graph construction.
- **Model Architecture:**
  We will primarily rely on a GAT/GATv2 model since it computes attention coefficients for each edge, while using GraphSAGE as a graph-based baseline and various non-graph baselines (can include XGBoost, simple regression, etc.). While GAT/GATv2 will be the core model used on each of the ideas below, we will choose to further experiment on some ideas with models like a THGNN or an R-GCN to see if they outperform the standard GAT model in the particular context of the idea.
- **Target & Loss Function:**
  The target will be a cross-sectional ranking of each node (tradeable asset) for each particular time t, where the primary/baseline loss function will be ListMLE due to its simplicity. As the project evolves, we can consider using more advanced loss functions that favor accuracy in the top-k rankings (which would be what we are interested in trading on) such as ApproxNDCG/NeuralNDCG, or directly use a differentiable Spearman coefficient loss function.
  Papers and literature in the field have shown that predicting exact return magnitudes frequently results in models that fit noise, thus resulting in either overfitting or over-generalizing, which makes our choice the best candidate for an efficient trading strategy.
- **Evaluation Metrics:**
  All models will be evaluated on a rolling-window, walk-forward basis using historical data over a reasonable time horizon (window will initially try 3-5 years, can adjust as necessary). For macro-focused models, an expanding window approach can also be employed for all data
  1. We will take the Spearman correlation coefficient between each model's predicted cross-sectional score and realized forward return at each evaluation step (roughly 1 week increments, though we can adjust up/down based on compute). Then, the change/decay of correlation over time can be observed to evaluate possible exit signals for constructing a trading strategy around it.
  2. Order the ranking as a list and split into deciles, then make sure that the top deciles consistently outperform by the greatest return down to the bottom decile performing the worst (make sure it has an actual cascading shape instead of returns per decile being centered around the average with minimal differences between each).
  3. Take some top-k of the stocks from the ranking and perform a portfolio backtest to gain insight into standard metrics like Sharpe, max drawdown, etc.
  4. Later on, once we have several different GNN constructions/themes, we can use ablations to more rigorously determine which features/themes from the constructions produce the most significant predictive power by evaluating different combinations/exclusions of features.

## Infrastructure:

1. Universe — the point-in-time tradeable universe plus a cross-source entity-resolution table (ticker/CUSIP/NAICS/GICS mapping). Build this first; everything else depends on it. For the dummy phase, this can just be the 11 GICS sector ETFs with trivial date coverage.
2. ✓ GraphSnapshot — the single most important interface decision in the whole system: a standardized data object holding a node feature matrix, edge index, edge weight array, node ID list, and a date. Every graph builder must output this exact shape, and every downstream layer must consume nothing else. Lock this schema before anyone writes real graph-building code — changing it later means re-touching every builder.
3. ✓ GraphBuilder (abstract base + dummy implementation) — a base class with one required method, build(date, universe) -> GraphSnapshot. Write a trivial SectorETFCorrelationBuilder subclass first (simple rolling correlation across the 11 sector ETFs) purely to exercise the pipeline end to end without needing BEA tables, segment data, or macro feeds to be ready.
4. ✓ FeatureEngine — shared node-feature computation (momentum, volatility, fundamentals, regime state), decoupled from any specific graph idea. Build and test against the dummy universe.
5. ✓ WalkForwardSplitter — generates train/test date-range folds with purge logic, operating purely on dates and independent of which graph idea is running. Unit test it against the dummy graph's date range to confirm fold boundaries and purge windows are computed correctly.
6. ✓ GraphSnapshotDataset — wraps a sequence of GraphSnapshot objects plus a WalkForwardSplitter fold into a trainable batch structure (PyTorch Geometric-compatible). This is the glue between steps 2/5 and the model layer.
7. ✓ ModelFactory and LossFactory — registries mapping config strings ("gatv2," "graphsage") and ("listmle," "spearman") to actual implementations, plus a shared Trainer class that takes a model, dataset, and loss and runs the training loop. Train on the dummy... sector graph purely to confirm the loop runs, losses decrease, and outputs have the right shape — the actual numbers will be economically meaningless at this stage, and that's fine.
8. ✓ Evaluator — computes rank IC and decile spread from scored outputs plus realized returns, logged to MLflow. Run it against the dummy graph's training output and manually sanity-check a couple of the IC calculations by hand to confirm the evaluator itself is correct before trusting it on real data.
9. Finally, build the three real GraphBuilder subclasses — CorrelationGraphBuilder, SupplyChainGraphBuilder, MacroGraphBuilder — in parallel, each producing the same GraphSnapshot schema from step 2.
10. ✓ BacktestEngine — build last, once real model outputs exist, converting rankings into portfolio weights/costs/tearsheet metrics.

Note: When actually running/using any part of the pipeline, ensure your changes are only to your local version of the config file for that piece of the pipeline, and do not change any of the actual code on the script itself. We should keep this use requirement in mind when developing each phase of the pipeline above.

## Supply Chain Graph

**Summary:**

This graph tests whether commodity price shocks propagate predictably through real, disclosed supply-chain relationships to affect equity returns — not just for firms with direct commodity exposure, but for indirectly exposed firms downstream, via multi-hop message passing through the production network. Rather than connecting every stock directly to every commodity, we restrict direct exposure to a small, systematically-defined set of firms and let the graph structure itself carry the shock further along real economic linkages.

**Node Set:**

Two node types:

1. Tradeable equities in our point-in-time universe
2. A small set of context-only commodity nodes (e.g., crude oil, gold, copper, aluminum) that are never scored, ranked, or traded — they exist purely to inject price signal into the graph

**Edge Construction:**

1. Commodity-to-firm edges (sparse, direct exposure only). We can classify direct exposure systematically using BEA's Input-Output Direct Requirements table (we can start with the "Detail" level, which has ~402 industries, or opt for a more frequently updated but less detailed "Summary" level. Link: https://apps.bea.gov/iTable/?reqid=1602&step=2&Categories=Core&isURI=1&_gl=1*1rpfj6p*_ga*MTE3MTQ5MDE1NS4xNzg4ODgzMzMx*_ga_J4698JNNFT*czE3ODg4ODMzMzEkbzEkZzEkdDE3ODg4ODM1MTAkajYwJGwwJGgw#eyJhcHBpZCI6MTYwMiwic3RlcHMiOlsxLDIsM10sImRhdGEiOltbImNhdGVnb3JpZXMiLCJBUiJdLFsiVGFibGVfTGlzdCIsIkN4SURSQVIiXV19 ), which reports how much of each commodity a given industry consumes as an intermediate input per dollar of output. Each firm's NAICS/SIC code (from Compustat, pulled point-in-time) is mapped to its corresponding BEA industry via BEA's published NAICS concordance . For firms with multiple business segments, we pull each segment's own NAICS classification from Compustat segment data and compute a revenue-weighted blend of each segment's BEA coefficient, rather than assigning one classification to the whole firm. A firm only receives a direct commodity edge if its (possibly blended) coefficient sits above a percentile threshold (e.g., top 10-15% of all industries for that commodity) — this keeps the direct-edge set genuinely sparse and restricted to real first-order dependency, rather than drawing a faint edge from every firm to every commodity.
   - Note: This has a good bit of data engineering overall
2. Firm-to-firm edges (supply chain backbone). Directional edges from Compustat segment and customer-supplier disclosure data, representing real economic input-output relationships between firms in our universe.

**Node and Edge Features:**

Commodity nodes carry their own daily price series (level and percent change) as node features. Direct commodity-to-firm edge weight is computed as the (static) BEA-derived coefficient multiplied by the (dynamic) realized daily commodity price change — so the edge weight itself moves daily even though the underlying industry-level structure only updates when BEA releases a new benchmark (roughly every 5 years for Detail-level data). Firm-to-firm edge weight is the standard input-output exposure percentage from segment data. Equity node features remain the standard set: momentum, volatility, fundamentals (value/quality/size), and anything else we find useful to add through research/brainstorming (like a volatility regime, etc).

**[Optional] NLP Layer:**

An alternative to the approach above for determining a company's exposure to direct commodity prices is to create a basic, preferably locally hosted NLP model that scans 10k filings, (specifically Item 1A risk factors and Item 7A quantitative/qualitative market-risk disclosures), to construct the exposures for each company individually instead of mapping them to their industry's average.

**Inputs:**

BEA Input-Output Direct Requirements table (Detail level, After Redefinitions); BEA NAICS-to-industry concordance; Compustat segment and customer-supplier data; point-in-time firm NAICS/SIC classifications; daily commodity price series; 10-K filings (Item 1A, Item 7A) for the refinement layer; standard equity fundamentals/technical panel for node features.

**Outputs:**

Rank IC and decile spread on the holdout period, evaluated on the same fixed horizon as all other graph ideas for fair comparison. A specific stress-test slice measuring whether firms 2+ hops from any commodity node show predictable return response to commodity price shocks — this is the cleanest direct test of whether multi-hop propagation is adding real signal beyond first-order exposure.

## Macro-Sensitivity Graph

**Summary:**

This graph tests whether a generic, statistically-estimated sensitivity to broad macro and cross-asset factors — interest rates, credit conditions, volatility regime, and key commodities — carries predictive information about equity returns, independent of any firm's actual disclosed economic dependencies. Unlike the supply chain graph, which restricts commodity exposure to a small, structurally-justified set of firms and propagates shocks through real linkages, this graph connects every stock directly to every macro factor via a broad estimated relationship, testing a competing hypothesis about how macro conditions reach equity prices.

**Node Set:**

Two node types:

- Tradeable equities in our point-in-time universe
- Context-only macro nodes, never scored or traded: Treasury yield level (e.g., 10-year yield), Treasury yield curve slope (2s10s spread) [optional], a credit spread index (IG or HY OAS), VIX, oil price (WTI/Brent), and gold price

**Edge Construction:**

Edges are dense rather than sparse — every equity node connects to every macro node, since macro sensitivity is diffuse across the universe rather than concentrated in a structurally-identifiable subset of firms. This also means we might have to restrict the equity universe if the compute cost becomes noticeably significant. Each edge weight is a rolling-window regression coefficient (beta) of the stock's return on the macro factor's change, recomputed each time the window advances.

**Node and Edge Features:**

Macro nodes carry their own daily time series (level and percent change) as node features. Equity node features remain our standard set: momentum, volatility, fundamentals (value/quality/size), and regime state. Edge weight is the magnitude of the rolling beta coefficient, consistent with our earlier decision to use magnitude rather than signed values for compatibility with the standard GATv2 pipeline — the sign of each beta will be logged separately as metadata for interpretability rather than fed into the model directly.

**[Optional] Temporal Architecture:**

Since macro sensitivities plausibly shift across regimes (e.g., a utility stock's rate-beta likely differs between a hiking cycle and a cutting cycle), we will test a THGNN-style temporal-relational architecture on this idea specifically, in addition to our GATv2 baseline, to see whether explicitly modeling how these sensitivities evolve over time outperforms a static snapshot approach.

**Inputs:**

Daily Treasury yield data (level and slope), credit spread index, VIX, oil and gold price series, plus our standard equity fundamentals/technical panel for node features. Note: For the equity universe, we can start with just the sector ETFs like XLF, XLE, etc.

**Outputs:**

Rank IC and decile spread on the holdout period, evaluated on the same fixed horizon as all other graph ideas. Attention-weight analysis identifying which macro factors the model weights most heavily, and whether that weighting shifts across different volatility/rate regimes....

## Phase 1: Static Correlation Graph

**Summary:**

This graph tests whether statistical co-movement between stocks, computed as a fixed structure over a rolling window and controlling for market-wide risk, carries predictive information about cross-sectional returns. It serves two purposes in our roadmap: as a standalone graph idea in its own right, and as the structural scaffolding that Phase 2 (the dynamic lead-lag network) narrows its search space against, rather than searching for lead-lag relationships across the full universe from scratch.

**Node Set:**

Tradeable equities in our point-in-time universe only.

**Edge Construction:**

Edges are computed via partial correlation (using graphical lasso) between residualized daily returns over a rolling window (initial target: 60-90 trading days), where returns are residualized against the market factor first so the graph captures co-movement not already explained by systematic risk. This is a symmetric relationship — the edge between stock A and stock B carries one weight, with no notion of directionality or timing offset. The graph is recomputed each time the rolling window advances, consistent with our shared rebalancing cadence, and slots directly into the same GATv2 pipeline used for our other precomputed-graph ideas.

Note: unlike the lead-lag network, this graph is built entirely by an external preprocessing step before the model ever runs — the graph-builder function and the GNN model are fully decoupled, so this idea reuses our shared codebase without modification.

**Node and Edge Features:**

Equity node features are our standard set: momentum, volatility, fundamentals (value/quality/size), and regime state. Edge weight is the partial correlation coefficient itself, specifically its absolute magnitude (treating strength of co-movement as what matters regardless of sign).

**Extension to Lead-Lag Network:**

Once this graph is running, we will extract each stock's top-N most correlated neighbors and use that list as the candidate leader pool for the Phase 2 lead-lag detector, rather than having that model search over all possible pairs in the universe. This reduces Phase 2's compute cost substantially and gives us a natural checkpoint: Phase 1's holdout rank IC becomes the baseline that Phase 2 must beat to justify the added architectural complexity and separate implementation track of the dynamic lag-detection module.

**Inputs:**

Daily price/return panel only.

**Outputs:**

Rank IC and decile spread on the holdout period, evaluated on the same fixed horizon as all other ideas. The top-N neighbor lists per stock, exported specifically as an input for the Phase 2 lead-lag network's candidate-pool restriction.

## Phase 2: Dynamic (Learned) Lead-Lag Network

**Summary:**

This graph tests whether directional, time-lagged information flow between stocks — one stock's price move systematically anticipating another's by a specific number of days — carries tradeable signal beyond what a standard contemporaneous correlation graph captures. Rather than fixing "who is related to whom" once per rolling window, the model relearns each stock's leaders and their specific lag values fresh every trading day, directly optimized against the prediction objective rather than computed from a separate statistical formula.

**Node Set:**

Only the tradeable equities in our universe (no commodities or external factors).

**Edge Construction:**

Edges are not precomputed on a fixed schedule; they are detected fresh each trading day by a cross-attention module. Each stock's rolling window of price/volume features is passed through a temporal encoder (e.g., an LSTM), producing a hidden state at every timestep. For a given target stock (the "lagger"), its most recent hidden state acts as a query against every other stock's hidden states from several recent lags (candidate keys). The model scores every (candidate leader, candidate lag) combination via attention and selects the top-k highest-scoring pairs as that stock's leaders for that specific day — this produces a directed, sparse, genuinely dynamic graph rather than a static adjacency matrix.

Note: This is architecturally distinct from our GAT pipeline used elsewhere — it's a purpose-built detection-and-prediction module rather than message passing over a fixed graph — so it will need its own implementation track rather than reusing the shared model code.

**Node and Edge Features:**

Node features are raw price/volume series (intraday high/low/open relative to close, daily return, log-volume, turnover), encoded into a temporal hidden state rather than used as static snapshot features. Edge weight is the attention score assigned to each selected (leader, lag) pair. Edge lag is itself a learned attribute unique to this graph — no other idea in our lineup has this concept, since correlation and supply-chain edges represent a single static relationship, not an explicit time offset.

**[Optional] Correlation-Graph Candidate Pool:**

An alternative to searching over all possible stock pairs for lead-lag relationships is to restrict each stock's candidate leader pool to its top-N most correlated names from our existing correlation graph (Idea 1), rather than searching the full universe. This reduces compute meaningfully and gives a cleaner interpretability story — "these stocks are correlated, and here's the specific lead-lag timing hiding inside that correlation" — at the cost of potentially missing a genuine lead-lag pair that happens to have low overall correlation.

**Inputs:**

Daily price/volume panel only (open/high/low/close, volume, turnover).

**Outputs:**

Rank information coefficient (IC) and decile spread on the holdout period, on the same fixed evaluation horizon as all other graph ideas. Evaluate an ablation against a static correlation graph to test whether dynamically-detected, directional lag relationships outperform a fixed, symmetric, contemporaneous correlation structure. A secondary analysis of the distribution of detected lag values across the universe (e.g., are most leader relationships 1-day, or do longer multi-day lags show up meaningfully), which informs whether our rebalancing frequency is even fast enough to act on this signal before it decays.

## Earnings & Information Spillover Graph

**Summary:**

This graph tests whether earnings surprises and narrative shocks (in news, filings, and earnings calls) propagate predictably across firms that share informational linkages, and whether modeling these spillovers as a graph improves cross-sectional ranking versus a non-relational baseline. Rather than relying solely on price correlation or macro betas, this construction treats co-coverage in text, similarity in risk language, and explicit earnings events as first-class relational objects that connect firms within and across industries. By introducing heterogeneous nodes for events and themes, the graph can capture both direct competitor responses and broader sector/regime effects around key information releases (e.g., earnings seasons, regulatory announcements).

**Node Set:**

Two primary node types are used:

1. Tradeable equities in our point-in-time universe, identical to other constructions so evaluation remains comparable.
2. Context-only event nodes representing discrete earnings releases and major news events, which are never scored, ranked, or traded and exist purely to inject information into the graph.

Optionally, we add a third node type for high-level topics (e.g., "regulatory risk", "AI exposure", "commodity sensitivity"), learned via topic modeling or embeddings over filings and news, to represent abstract themes that link firms and events without introducing additional tradeable assets.

**Edge Construction:**

Stock–event edges connect each firm to its own earnings events (quarterly reports, pre-announcements) with edge features derived from structured data: earnings surprise relative to consensus, changes in forward guidance, event-window abnormal returns, and realized volatility around the event. Competitor stock–event edges connect earnings events of one firm to peer firms in the same industry or custom peer sets (e.g., GICS/NAICS clusters, product-based competitors), with weight encoding how relevant the event is to each peer via similarity in business mix, customer base, or historical co-movement around earnings. Separately, stock–stock edges are built from text-based relationships: co-mentions in news over rolling windows, semantic similarity between 10‑K risk-factor sections, and similarity between earnings-call transcripts (e.g., shared topics, sentiment patterns). Edge weights combine recent co-mention frequency with embedding-based similarity scores so that firms frequently discussed together, or with similar narrative risk profiles, are strongly linked even if their fundamentals or macro betas differ....

**Node and Edge Features:**

Equity node features reuse the shared FeatureEngine: momentum, volatility, fundamentals (value/quality/size), regime-state indicators, and any additional technical or fundamental signals developed elsewhere in the project. Event nodes carry structured and unstructured features: surprise magnitude, guidance changes, realized abnormal return, plus sentiment and topic distributions from the associated press release and earnings-call transcript.

Edge features enrich the relational structure: for stock–event edges, they include event type (earnings vs guidance vs regulatory news), surprise direction (beat/miss), and intensity; for stock–stock text edges, they include co-mention counts, document-similarity scores, sentiment correlation, and topic-overlap metrics. Topic nodes, when used, have features summarizing their own time-varying prevalence and average sentiment across the universe, while edges from topics to stocks/events carry the strength of association between each entity and the theme.

**Inputs:**

Historical earnings data with analyst estimates and realized surprises, point-in-time consensus forecasts and guidance changes, and intraday or daily price data to compute event-window abnormal returns. Text inputs consist of 10‑K filings (especially Item 1A risk factors and Item 7/7A MD&A and market-risk sections), earnings-call transcripts, and news mapped to tickers (can test the MassiveAPI "News" feature). Additionally, we use any existing universe and entity-resolution tables from the Infrastructure (ticker/identifier mapping, GICS/NAICS classifications, peer sets) to define competitor relationships consistently.

Optional inputs for topic nodes include pre-trained language models or topic models (e.g., BERT embeddings or LDA) applied to filings, transcripts, and news to derive theme representations. All time series are aligned to the same daily or weekly calendar as other graph constructions so that GraphSnapshot objects remain consistent with the shared pipeline design.

**Outputs:**

As with other ideas, the primary outputs are rank information coefficient (Spearman correlation between predicted cross-sectional scores and realized forward returns) and decile spreads on a holdout period, computed on the same fixed horizon as the other graph constructions for fair comparison. We also generate portfolio backtests for top‑kkk names to evaluate Sharpe, drawdown, and turnover, focusing on windows around earnings seasons to test whether the information graph adds incremental signal during these high-news periods.

Secondary outputs include diagnostics specific to this construction: how peers respond to focal earnings surprises (conditional abnormal returns of competitors), attention-weight patterns over event nodes during different regimes, and topic-level analyses showing which themes the model relies on most heavily when ranking stocks before and after major information shocks.

## Variations and Extensions:

**Text-only variant:** one extension restricts the graph to text-driven stock–stock and stock–topic edges, omitting earnings event nodes entirely to test whether narrative similarity alone carries predictive power. In this variant, the model ranks equities based on static and time-varying textual relationships (filings, transcripts, news) plus standard node features, providing a clean baseline for "information without explicit events."

**Earnings-only variant:** another extension focuses purely on earnings-event and competitor edges, treating each earnings season as a local information cascade and evaluating prediction performance on short horizons (e.g., 5–20 trading days post-event). Here, we examine whether modeling explicit event spillovers among peers outperforms simpler rules like "buy competitors of big positive surprise firms," and whether the learned attention over event nodes reveals systematic patterns in how shocks propagate across industries.

**Hybrid narrative–event graph:** the full construction combines both layers, allowing earnings events to inject shocks that then propagate through text-based similarity and topic nodes, capturing cases where certain narratives amplify or dampen the impact of earnings surprises across the broader universe. We can run sub-variants that remove event nodes but keep text edges, remove text features but keep structured earnings edges, or drop topic nodes, to identify which components drive most of the predictive power and whether the added complexity is justified relative to simpler constructions.
