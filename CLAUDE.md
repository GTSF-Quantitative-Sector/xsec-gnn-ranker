# gnns — cross-sectional GNN ranker

## Project

Test whether a GNN that scores each stock conditioned on its own features and its graph
neighbors produces a cross-sectional ranking signal that outperforms a non-relational
factor model, evaluated on rank IC, decile cascade, and a long-only top-k backtest. See
`docs/project_overview.md` for the full plan.

## Rules

- Users change config, not code. Behavior that's meant to be tunable goes into `config/config.yaml`.
- Dependencies are added to `environment.yml` only when code first imports them.
- Every pipeline run logs its resolved config and the git SHA to MLflow.
- Every user-facing command or setup step is documented in `README.md` in the same change
  that adds it.

## Tooling

Conda env `gnns`, Python 3.12. Install with `conda env create -f environment.yml`.

- **pyright** is the type/import safety net. The `pyright-lsp` plugin gives me diagnostics
  automatically after each edit; the env must be active when you start Claude Code so pyright
  resolves imports correctly.
- **ruff** is style and lint. Run `ruff check` once before each commit.
- **pytest** for tests.

## Layout

The `src/gnns/` package is split into four subpackages. Each has its own `CLAUDE.md` with
that component's contracts, decisions, and open questions, which auto-loads when I touch
files in that folder.

| Folder | Contains |
| :-- | :-- |
| `src/gnns/data/` | Universe, FeatureEngine, GraphSnapshot schema |
| `src/gnns/graphs/` | GraphBuilder abstract class + concrete builders |
| `src/gnns/train/` | WalkForwardSplitter, Dataset, Model/Loss factories, Trainer |
| `src/gnns/eval/` | Evaluator, Backtest |

Top-level modules (added when needed): `config.py` (loads `config.yaml` + `machine.yaml`),
`cli.py` (entry point). Tests in `tests/` mirror this layout. Config files are
`config/config.yaml` (tracked) and `config/machine.yaml` (gitignored).

## Where to look

| File | For |
| :-- | :-- |
| `docs/project_overview.md` | the full research plan |
| `docs/PLAN.md` | current pipeline status: what's built, what's next |
| `docs/CONTRACTS.md` | locked interfaces (GraphSnapshot, config schema) |
| `docs/RESEARCH_PROTOCOL.md` | walk-forward, purge/embargo, MLflow conventions |
