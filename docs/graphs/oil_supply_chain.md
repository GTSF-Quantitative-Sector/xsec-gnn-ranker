# Oil supply-chain graph

Does a graph organized around crude supply, refining, and transport fuel improve five-session equity rankings over an MLP with the same equity features and observed Brent proxy moves? This prototype uses economic exposure categories fixed before model fitting. It does not estimate contractual links, weights from purchasing volumes, or the causal effect of an oil shock.

## Universe and economic relationships

| Node or edge | Rationale |
| --- | --- |
| **BNO** | Context node for Brent. Its market price reflects an exchange-traded fund targeting short-term Brent futures. Contract rolls, fund costs, and futures basis can separate BNO returns from Brent spot and from US jet fuel or diesel prices. [BNO 2022 Form 10-K](https://www.sec.gov/Archives/edgar/data/1472494/000141057823000169/bno-20221231x10k.htm) |
| **BNO → XOM, CVX, COP** (`producer_revenue`) | Higher realized crude prices can support upstream revenue. XOM and CVX also buy crude and refine or market products, so their consolidated exposures can differ from a producer such as COP. [ExxonMobil](https://www.sec.gov/Archives/edgar/data/34088/000003408823000020/xom-20221231.htm), [Chevron](https://www.sec.gov/Archives/edgar/data/93410/000009341023000009/cvx-20221231.htm), [ConocoPhillips](https://www.sec.gov/Archives/edgar/data/1163165/000116316523000006/cop-20221231.htm) |
| **BNO → MPC, VLO** (`crude_feedstock`) | Crude is a refinery input cost. Product prices, crude grades, feedstock mix, and operating costs also affect margins. The graph does not fix a return sign for either refiner. [Marathon Petroleum](https://www.sec.gov/Archives/edgar/data/1510295/000151029523000012/mpc-20221231.htm), [Valero](https://www.sec.gov/Archives/edgar/data/1035002/000103500223000027/vlo-20221231.htm), [EIA refinery inputs and outputs](https://www.eia.gov/energyexplained/oil-and-petroleum-products/refining-crude-oil-inputs-and-outputs.php) |
| **MPC, VLO → DAL, UAL, UPS** (`jet_fuel`) | Jet fuel connects refining economics to airline and parcel-air operating costs. Delta's Monroe refinery can offset some exposure to the refining margin. UPS also consumes ground fuel and can pass through part of its exposure via surcharges. UPS appears once in the downstream graph, via its air channel. [Delta](https://www.sec.gov/Archives/edgar/data/27904/000002790423000003/dal-20221231.htm), [United](https://www.sec.gov/Archives/edgar/data/100517/000010051723000048/ual-20221231.htm), [UPS](https://www.sec.gov/Archives/edgar/data/1090727/000109072723000006/ups-20221231.htm) |
| **MPC, VLO → ODFL** (`road_fuel`) | Diesel is an operating input for Old Dominion. Fuel surcharge programs can mitigate the impact, and timing and freight demand can change the net effect. [Old Dominion](https://www.sec.gov/Archives/edgar/data/878927/000095017023003783/odfl-20221231.htm) |
| **NEE** | Ranked utility comparison without an oil-supply edge. FPL's generation mix and regulated fuel cost recovery give it a different exposure from transport companies. NEE can still co-move with energy markets and receives the same BNO features as the other equities. [NextEra Energy and FPL](https://www.sec.gov/Archives/edgar/data/37634/000075330823000019/nee-20221231.htm) |

The refiner-to-carrier edges represent industry pathways through the refined-products market. The filings support the input and cost exposures. They do not establish that either named refiner supplied a specific carrier. The cited 2022 filings substantiate the broad roles before the holdout but were published after some search dates. Treat the fixed historical topology and present-day ticker selection as retrospective. A vintage-aware universe and timestamped supplier disclosures are needed for point-in-time company-link claims.

## Method

- The graph has ten ranked equity nodes and one non-target oil node when the full panel is available. Thirteen directed edges use unit weights and one-hot categories. No exposure sign or strength is hard-coded. Two message-passing layers let information travel from BNO through a refiner to a carrier.
- Equity inputs are 21- and 63-bar momentum, 21-bar volatility, and BNO returns over 1, 5, and 20 bars. The BNO node carries those returns in its own column block. Both rankers see the same equity-level inputs. BNO returns bypass cross-sectional normalization so that a shared move retains its magnitude.
- The graph builder only reads bars known by the signal close. An equity needs a bar on that date. The oil calculation requires a bar at each exact endpoint on the common bar axis, without forward-filling. Edges with missing nodes are omitted. If BNO is unavailable or no equities qualify, the snapshot is skipped.
- The existing label enters at the next open and exits after the configured holding horizon. Search-period labels must exit before the holdout. Backtests still require a coverage audit, since missing future labels can remove names from the ranked output and skipped dates leave gaps in the simulated holdings.

## Run locally

Use the repo's `gnns` environment and the Massive data pipeline. In the optional `config/machine.yaml`, select this graph and include the oil universe and benchmarks. The entire `data` block is overridden by this local setting.

```yaml
graph: oil_supply_chain
data:
  universes: [oil_supply_chain, benchmarks]
  bar: 1d
  start: 2010-01-01
  end: 2026-06-30
```

```bash
gnns build-data
gnns build-graphs
pytest tests/graphs/test_oil_supply_chain.py
gnns train
```

The data build requires a Massive API key with historical access for the full ticker list. An existing compatible Parquet and manifest can be used instead. The dataset needs enough history before the existing 2024 holdout boundary for the 63-bar warm-up and walk-forward windows. Keep at least the holdout anchor session in the file so the splitter can place the boundary. The graph reuses the existing universe manifest format and loader.

## Findings and follow-up

The test suite passed 194 tests, including graph construction, BNO's target mask, invariance to truncating future bars, the search cutoff, and CPU training for both models. A separate synthetic CLI smoke run built 43 snapshots with 11 nodes and 13 edges each. It trained GATv2 and the MLP for two epochs on one search fold and verified 16 top-level MLflow artifacts, including both prediction tables, the backtest folder, and report charts. The run used generated data and temporary storage to verify the pipeline. There is no market-data result because the checkout had no price files or Massive credential.

After auditing real price coverage, run several seeds on search folds. Compare rank IC and cost-adjusted results against the matched MLP. Predefine a topology control that rewires edges while preserving degrees, and compare one-hop and two-hop variants on the same dates. Separate jet fuel, diesel, and refinery crack-spread series could help distinguish input cost transmission from broad oil-market co-movement. Timestamped disclosures and BEA input-output data would support a second prototype with measured links.
