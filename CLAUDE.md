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

## Workflow

GitHub Issues are the only to-do list; the commands are in README ("Workflow").

- A workstream is one parent issue, one branch `<N>-<slug>` and one PR that says
  `Closes #<N>`. Its steps are the parent's checklist: tick each as it's done. You may
  create, edit and comment on issues, push your branch and open the PR. Never merge, push
  to `main`, close a parent issue, or edit another workstream's checklist.
- Every issue has one area label: infra, data, graphs, train, eval, backtest. Blockers
  only point upstream, in that order.
- **Needing another workstream's work:** open a sub-issue under its parent (standalone,
  with the area label, if no open workstream owns it) with Need, Where it's used,
  Constraints, Evidence and Done when, and mark your issue blocked by it
  (`gh issue edit <N> --add-blocked-by <M>`). Finish your independent work, then stop and
  report "blocked on #M": no workarounds, no waiting in the background.
- **Owning a sub-issue:** design the solution and, before opening your PR, comment on the
  sub-issue starting with `## Resolution`, with these sections, each non-empty:
  `### What changed`, `### Where` (paths and the PR, not commit SHAs), `### How to use it`
  (the interface: signatures, columns, null rules, an example), `### Done when` (each item
  of the sub-issue and what shows it), `### Differs from the request` ("None" if nothing).
  Your PR also says `Closes #<sub-issue>` for each one.
- **Resuming after a blocker closed:** rebase on `main`, then read the blocker's Resolution
  and the merged code against your sub-issue. If anything is missing or differs from what
  you need, stop and tell the user.
- **Cycles:** if the issue check reports one, ship the sub-issue it names first, on its own
  branch `<M>-<slug>` from `main` with its own PR. Label it `ship-first` and comment the
  order on each parent in the cycle.
- The issue check (`python scripts/issue_check.py check`) lists what still blocks your PR.
  It runs at session start, after each commit and in CI.
- A PR that finishes one of the ten Infrastructure steps in `docs/project_overview.md`
  marks that step ✓.

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
| GitHub Issues (`gh issue list`) | what's in progress and what's next |
| `docs/CONTRACTS.md` | locked interfaces (GraphSnapshot, config schema) |
| `docs/RESEARCH_PROTOCOL.md` | walk-forward, purge, MLflow conventions |
