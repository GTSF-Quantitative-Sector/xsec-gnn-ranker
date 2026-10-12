# gnns.graphs

The GraphBuilder abstract base class and all concrete builders. Every builder's
`build(date, universe)` must return a GraphSnapshot (see `src/gnns/data/`). First
implementation is a dummy sector-ETF correlation builder that exercises the pipeline
end to end.
