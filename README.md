# xsec-gnn-ranker

Cross-sectional GNN stock ranker (GTSF Quant Sector). Research plan:
[docs/project_overview.md](docs/project_overview.md) (✓ marks finished steps). In progress
and next: the repo's [Issues](https://github.com/GTSF-Quantitative-Sector/xsec-gnn-ranker/issues).

## Setup

    conda env create -f environment.yml
    conda activate gnns

After pulling a change to `environment.yml`, run `conda env update -f environment.yml --prune`.

Once per clone, turn on the repo's git hooks (worktrees and slots share the setting):

    git config core.hooksPath .githooks

Install the GitHub CLI ([cli.github.com](https://cli.github.com)) and log in once:
`gh auth login`. The issue check, its hooks and `slot.py sync` use it.

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

### CI

GitHub Actions (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`,
`pyright` and `pytest` on Ubuntu for every push to `main` and every pull request, in an
env built from `environment.yml`. Results are on the repo's Actions tab and on each pull
request. Run the same four commands locally before pushing.
On pull requests a second job, `issues`, runs the issue check below.

Merge pull requests with **Merge when ready** (the merge queue). GitHub then reruns both
jobs on the latest `main` plus the PR and merges only if they pass, so the issue check
sees the issues as they are at merge time, not as they were at the PR's last push. A
merge takes a few minutes.

### Workflow

GitHub Issues are the to-do list, and agents follow the rules in `CLAUDE.md` ("Workflow").

1. **Start:** open an issue with an area label (infra, data, graphs, train, eval,
   backtest) and a Goal, Done when and Steps checklist, then
   `python scripts/slot.py start <slot> <N>-<slug>` and start a session there.
2. **Work:** the agent ticks steps as it goes. When it needs another workstream's work,
   it opens a sub-issue there, marks its own issue blocked, and stops once only blocked
   work is left.
3. **Finish:** the agent pushes the branch and opens a PR saying `Closes #<N>` (plus
   `Closes #<sub-issue>` for each handoff it resolved). CI runs `check` and `issues`.
4. **Merge:** **Merge when ready** (below), then `python scripts/slot.py sync`, and resume
   the threads it lists as "resume".

### Issue check

`scripts/issue_check.py` checks the GitHub issue of the current branch (`<N>-<slug>` works
on issue #N) and lists what still blocks its pull request. It needs `gh`, logged in
(`gh auth login`).

    python scripts/issue_check.py check        # what still blocks this branch's PR
    python scripts/issue_check.py check --ci   # same, but exit 1 if anything does (CI)
    python scripts/issue_check.py session      # the issue, its handoffs and blockers, the check

Claude Code runs `session` at the start of every session (`.claude/settings.json`), so each
agent starts with its workstream's issue, open handoffs and blockers in context.
After each commit, the `post-commit` git hook runs `check` and prints its findings (it never
blocks the commit).

### Working on several branches at once (slots)

The repo folder (home) stays on `main`. Parallel tasks run in **slots**: reusable git
worktrees next to it (`<repo>-slot1`, `-slot2`, …), each with its own conda env
(`gnns-slot1`, …) and the package installed from that folder, so a slot always runs its
own code. Slots share the home repo's `data/` (via their `machine.yaml`) and copy its `.env`.
MLflow runs stay per slot. From the home repo, with conda available:

    python scripts/slot.py setup 1 2 3          # once: worktree + env clone (~2 min each)
    python scripts/slot.py start 1 backtester   # branch from local main (seconds)
    python scripts/slot.py list                 # slots, branches, uncommitted work
    python scripts/slot.py sync                 # after a merge: update main, free slots
    python scripts/slot.py remove 1             # delete a slot's worktree and env

`start` refuses if the slot has uncommitted work, reuses the branch if it already exists,
and updates the slot's env only when `environment.yml` changed. Pull `main` in the home
repo first. Then start the session from a prompt with the slot's env active (`conda
activate gnns-slot1`, `cd` into the slot, `claude`); if you use VS Code, close it first so
the new window picks up the slot's env. Finish a task by pushing its branch and opening a
pull request.

After merging a pull request, run `sync` (needs `gh`). It pulls `main` in the home repo,
deletes each branch whose PR merged and detaches its slot at `main` (it keeps a branch with
commits after its PR, or a slot with uncommitted changes), then lists each slot's thread:
blocked on an open issue, resume (a blocker's fix is on `main` but not yet in the branch:
rebase, then read the blocker's Resolution), or working.

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

1. Pick a construction in `config/config.yaml` with `graph: <name>`. Everything specific to
   it lives in `config/graphs/<name>.yaml`: builder, universe, `params`, its own node
   `features`, and later `sources`:

       builder: sector_etf_correlation
       universe: sector_etfs
       params:
         window_bars: 60
       features:
         min_history_bars: 60      # a node needs this many bars before it is used
         list:
           - {name: momentum, bars: 20}
           - {name: momentum, bars: 60}
           - {name: volatility, bars: 20}

2. Settings shared by every construction stay in `config/config.yaml`, e.g.
   `horizon_bars: 5`: each label is the return over the next 5 bars, and there is one
   snapshot (and one rebalance) every 5 bars, ≈ weekly on daily data.

3. Run `gnns build-graphs`. It builds every snapshot in memory and prints a summary
   (count, date range, node and edge counts); nothing is written to disk.

`min_history_bars` must cover the builder's price window (e.g. `window_bars`); the command
stops with an error otherwise.

### Train, evaluate and plot

Needs the construction's universe and the `benchmarks` universe built (`gnns build-data`).

1. In the construction's graph file, set the `model` (`gatv2`, `graphsage` or `mlp`) and
   its matched no-graph `baseline` (normally `mlp`), plus `extra_model_features` and
   `model_edge_attrs` if a builder adds columns the model should see.
2. Shared settings in `config/config.yaml`: `train` (loss, learning rate, batch size,
   epochs, early-stopping patience, seed), `eval` (quantiles, permutation repeats,
   rolling-IC window), `backtest` (see "Backtest") and `normalize`.
3. Run `gnns train`. It trains the model and the baseline on every search fold (held-out
   folds are never touched), stopping early on validation rank IC, scores each fold's
   test period, and logs an MLflow run: resolved config, git SHA, dataset manifest,
   test IC metrics, validation IC (`<model>_val_ic`, the column to compare runs on),
   training histories and predictions. It then backtests the run (see "Backtest") and
   makes its charts (step 5), and prints the run id with a backtest summary.
4. Browse runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db` (open
   http://127.0.0.1:5000). Runs live in the gitignored `mlflow.db` and `mlruns/`.
5. Charts: `gnns train` makes them for its run. To remake them for any run (e.g. one
   trained before a chart existed), run `gnns report <run_id>`. It works from that run's
   saved predictions (no retraining, a few seconds) and attaches the charts to the run:
   rank IC over time, cumulative IC, IC by fold (model vs baseline), quantile cascade,
   quantile equity curves, score spread, permutation importance per
   input and for edges (`eval.permutation_repeats`), per-fold training curves with the
   no-information loss line, and the feature correlation matrix (search period).

### Backtest

Needs the `benchmarks` universe built with the `backtest.benchmark` and
`backtest.risk_free` tickers (SPY and BIL in `config/universes/benchmarks.txt`).

1. Set the `backtest` block in `config/config.yaml`:

       backtest:
         top_k: 3              # names held at each rebalance, equal weight
         weighting: equal
         cost_bps: [0, 3, 6]   # per side, on traded value; every level is reported
         benchmark: SPY
         risk_free: BIL

2. `gnns train` backtests every run it logs (and skips with a message if the benchmarks
   are not built). To backtest a run again, e.g. after changing `backtest:`, run
   `gnns backtest <run_id>`: it uses the run's saved predictions, horizon and bar with the
   current `backtest:` settings, and replaces the run's backtest metrics, `backtest.*`
   tags and `backtest/` artifacts.

Terms: the horizon (`horizon_bars`) is a length; a rebalance is one snapshot date where
the portfolio is set, then held for one horizon until the next rebalance, so each
rebalance's return is one label. "Per rebalance" figures and the rows of
`rebalances_<model>.parquet` count these. "Period" is kept for a fold's train /
validation / test periods (and the search and held-out periods).

At each rebalance the portfolio holds the `top_k`
highest scores at equal weight, entering at the next bar's open like the labels, and
trades back to those weights (turnover = sum of |target - drifted weight|; the first
rebalance buys from cash), paying `cost_bps` per side on the traded value. Metrics per
model and cost level (suffix `_<bps>bps`): `ann_return`, `ann_vol`, `sharpe` (over the
risk-free return), `max_drawdown`, `max_dd_days` (longest time below a previous peak, in
trading days), `excess_return`, `tracking_error`, `ir` (vs the benchmark), `win_rate`,
`win_rate_vs_benchmark`, `cost_drag`; per model: `turnover` (one-way, per rebalance),
`turnover_annual`, `break_even_bps` (the cost at which the portfolio stops beating the
benchmark). Artifacts in `backtest/`: equity, drawdown and per-year charts (each year's
tick shows its rebalance count n, so partial years stand out), a per-rebalance table per
model (`rebalances_<model>.parquet`) and `backtest.yaml`. A held name without a
forward return stops the backtest with an error.

If you shrink `cost_bps`, metrics for levels you removed stay on the run from the earlier
backtest; the `backtest.cost_bps` tag lists the current levels.
