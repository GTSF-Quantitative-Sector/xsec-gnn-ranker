# xsec-gnn-ranker

Cross-sectional GNN stock ranker (GTSF Quant Sector). Research plan:
[docs/project_overview.md](docs/project_overview.md). Current status: [docs/PLAN.md](docs/PLAN.md).

## Setup

    conda env create -f environment.yml
    conda activate gnns

After pulling a change to `environment.yml`, run `conda env update -f environment.yml --prune`.

Always run the pipeline from an activated `gnns` env: torch, cuDNN and MKL find their
libraries through the env's `PATH`. `conda` installs the CUDA build of pytorch when it finds
an NVIDIA driver, and the CPU build otherwise.

**Windows, once per machine, before creating the env:**

- Enable long paths (pytorch's package has paths over 260 characters), in an
  administrator PowerShell:

      New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force

- If `conda config --show channels` lists `repo.anaconda.com` URLs, run
  `conda config --set channel_priority strict`. Otherwise conda can install Anaconda's
  pytorch next to conda-forge's cuDNN, and torch fails with "Cannot load symbol
  cudnnGetVersion".

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

### Build graph snapshots

Needs the construction's universe built first (`gnns build-data`).

1. Pick a construction in `config/config.yaml` with `graph: <name>`. Its settings live in
   `config/graphs/<name>.yaml` (builder, universe, `params`, and later `sources`):

       builder: sector_etf_correlation
       universe: sector_etfs
       params:
         window_bars: 60

2. Set the shared settings in `config/config.yaml`:

       snapshot_every_bars: 5      # one snapshot every 5 bars (≈ weekly on daily data)
       features:
         min_history_bars: 60      # a node needs this many bars before it is used
         list:
           - {name: momentum, bars: 20}
           - {name: momentum, bars: 60}
           - {name: volatility, bars: 20}

3. Run `gnns build-graphs`. It builds every snapshot in memory and prints a summary
   (count, date range, node and edge counts); nothing is written to disk.

`min_history_bars` must cover the builder's price window (e.g. `window_bars`); the command
stops with an error otherwise.
