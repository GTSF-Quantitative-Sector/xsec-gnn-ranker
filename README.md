# xsec-gnn-ranker

Cross-sectional GNN stock ranker (GTSF Quant Sector). Research plan:
[docs/project_overview.md](docs/project_overview.md). Current status: [docs/PLAN.md](docs/PLAN.md).

## Setup

    conda env create -f environment.yml
    conda activate gnns

After pulling a change to `environment.yml`, run `conda env update -f environment.yml --prune`.

Create `.env` at the repo root (gitignored) with your Massive API key:

    MASSIVE_API_KEY=your_key_here

Per-machine settings go in `config/machine.yaml` (gitignored, optional):

    data_root: data   # where datasets are written (default: data/)

## Usage

Every stage runs through the `gnns` command and reads `config/config.yaml`. To change what a
stage does, edit config, not code.

### Build price datasets

1. Choose universes, or add one as `config/universes/<name>.txt` (one ticker per line).
   Keep benchmark tickers in the `benchmarks` universe, never in a construction's list: every
   ticker in a universe becomes a ranked graph node.
2. Set the `data` block in `config/config.yaml`:

       data:
         universes:
           - sector_etfs
           - benchmarks
         bar: 1d            # shared by every universe; only 1d is supported so far
         start: 2010-01-01
         end: 2026-06-30    # inclusive

3. Run `gnns build-data`.

This writes `<data_root>/<universe>/<bar>/prices.parquet` and `manifest.json` for each
universe (schema in [docs/CONTRACTS.md](docs/CONTRACTS.md)). Prices are adjusted for splits
and dividends as of the build date, so every build re-downloads the full range. All universes
are validated before any is written; if one fails, the build stops with a list of problems
and leaves every existing dataset untouched.

To use a dataset in code: `gnns.data.store.load(root, universe, bar, start, end)` returns a
polars `LazyFrame`.
