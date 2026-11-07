"""`gnns sweep`: two-stage grid search on the search folds (design: GitHub issue #9).

Stage 1 trains the GNN (`model`) on every `model_grid` combination x seed, building the
data once per distinct data-side setting. Stage 2 fixes the data side at the stage-1
winner and trains the baseline on every `baseline_grid` combination x seed, on the
winner's data (no rebuild). Each stage's winner is its combination with the highest mean
validation IC over seeds; ties go to the earlier combination in grid order.

Stage runs log validation metrics only. One final run per seed pairs the two winners
from results kept in memory (no retraining), with test predictions and permutation
importance; `gnns sweep` backtests and charts those. All are children of one parent run
holding the spec, the results table, the summary and `frozen.yaml`. Which keys each grid
may sweep: docs/RESEARCH_PROTOCOL.md, "Settings: tuned vs fixed".
"""

import copy
import itertools
import json
import logging
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import polars as pl
import torch
import yaml
from mlflow.entities import Param
from torch_geometric.data import Data

from gnns.config import git_sha
from gnns.data.features import FEATURES
from gnns.graphs.build import BUILDERS
from gnns.train import run
from gnns.train.dataset import model_columns
from gnns.train.losses import LOSSES
from gnns.train.models import build_model

log = logging.getLogger(__name__)
PROTOCOL = 'docs/RESEARCH_PROTOCOL.md, "Settings: tuned vs fixed"'
SHARED = ("params", "features", "extra_model_features")  # data side: both models see it
GRIDS = {  # grid -> (role it tunes, graph-file sections it may sweep)
    "model_grid": ("model", (*SHARED, "model", "model_edge_attrs")),
    "baseline_grid": ("baseline", ("baseline",)),
}
NESTED = {"params", "features", "model", "baseline"}  # sweep keys inside these, not them
DERIVED = "features.min_history_bars"
# Graph-file section order, so frozen.yaml reads like the file it is pasted into.
ORDER = ("params", "features", "extra_model_features", "model", "model_edge_attrs", "baseline")


@dataclass(frozen=True)
class Combo:
    index: int  # position in grid order
    overrides: dict[str, Any]  # path within the graph file -> value
    name: str  # e.g. "params.window_bars=40, features.list=#2" (list values by position)


@dataclass
class Best:
    """A stage's best combination so far, with what the final runs need."""

    combo: Combo
    cfg: dict  # resolved config with the combination applied (seed: the file's)
    score: float  # mean validation IC over seeds
    results: dict[int, run.Trained]  # seed -> trained model
    built: run.Built


@dataclass
class Swept:
    """What `sweep` returns: the parent run, the final runs and what to freeze."""

    parent_run_id: str
    final_run_ids: list[str]
    frozen: dict
    summary: dict = field(default_factory=dict)


def sweep(cfg: dict) -> Swept:
    """Run both stages and the final runs; log everything under one parent run."""
    spec = parse_spec(cfg)
    combos = {grid: expand(spec[grid]) for grid in GRIDS}
    history = max(required_history(apply(cfg, c.overrides)) for c in combos["model_grid"])
    base = apply(cfg, {DERIVED: history})
    for grid, cs in combos.items():
        for c in cs:
            check_combo(apply(base, c.overrides), grid)
    device = run.pick_device(cfg)
    seeds = spec["seeds"]
    n1, n2 = len(combos["model_grid"]), len(combos["baseline_grid"])
    log.info(
        "sweep: %d model and %d baseline combinations x %d seeds; min_history_bars %d",
        n1,
        n2,
        len(seeds),
        history,
    )

    client = mlflow.MlflowClient(run.mlflow_tracking_uri())
    exp_id = run.experiment(cfg["graph"]["name"])
    parent = client.create_run(exp_id, run_name="sweep", tags={"sweep.stage": "parent"})
    parent_id = parent.info.run_id
    try:
        rows: list[dict] = []
        best1 = _stage(1, combos["model_grid"], base, seeds, device, parent_id, rows, None)
        best2 = _stage(2, combos["baseline_grid"], best1.cfg, seeds, device, parent_id, rows, best1)
        final_cfg = apply(best1.cfg, best2.combo.overrides)
        finals, tests = _final_runs(final_cfg, best1, best2, seeds, device, parent_id)
        tuned = best1.combo.overrides | best2.combo.overrides | {DERIVED: history}
        frozen = _nest(dict(sorted(tuned.items(), key=lambda kv: ORDER.index(kv[0].split(".")[0]))))
        table = pl.DataFrame(rows)
        summary = _summary(table, best1, best2, seeds, tests, finals)
        _log_parent(client, parent_id, base, spec, table, summary, frozen, best1.built)
    except BaseException:
        client.set_terminated(parent_id, "FAILED")
        raise
    client.set_terminated(parent_id, "FINISHED")
    return Swept(parent_id, finals, frozen, summary)


def parse_spec(cfg: dict) -> dict:
    """The graph file's `sweep:` block, checked: seeds, and each grid's keys and values."""
    graph = cfg["graph"]
    if "sweep" not in graph:
        raise ValueError(f"config/graphs/{graph['name']}.yaml has no sweep: block (README, Sweep)")
    spec = graph["sweep"]
    if missing := sorted({"seeds", *GRIDS} - spec.keys()):
        raise ValueError(f"sweep: needs {missing}")
    if unknown := sorted(spec.keys() - {"seeds", *GRIDS}):
        raise ValueError(f"sweep: unknown keys {unknown}; valid: {['seeds', *GRIDS]}")
    seeds = spec["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(isinstance(s, int) for s in seeds)
        or len(set(seeds)) < len(seeds)
    ):
        raise ValueError(f"sweep.seeds must be a non-empty list of distinct ints, got {seeds!r}")
    for grid in GRIDS:
        if not isinstance(spec[grid], dict) or not spec[grid]:
            raise ValueError(f"sweep.{grid} must map at least one key to a list of values")
        for key, values in spec[grid].items():
            check_key(cfg, grid, key)
            if not isinstance(values, list) or not values:
                raise ValueError(f"sweep.{grid}.{key} must be a non-empty list of values")
    return spec


def check_key(cfg: dict, grid: str, key: str) -> None:
    """Raise unless `key` is a path in the graph file that `grid` may sweep."""
    where = f"sweep.{grid} key {key!r}"
    parts = key.split(".")
    section = parts[0]
    allowed = GRIDS[grid][1]
    if key == DERIVED:
        raise ValueError(f"{where}: derived, not tuned; the sweep sets it ({PROTOCOL})")
    if section not in allowed:
        other = next((g for g, (_, sections) in GRIDS.items() if section in sections), None)
        if section in SHARED:
            why = "the data side is tuned in stage 1 and fixed in stage 2; use model_grid"
        elif other is not None:
            why = f"belongs in {other}"
        else:
            why = "fixed for the whole experiment"
        raise ValueError(f"{where}: {why} ({PROTOCOL})")
    if section in NESTED and len(parts) < 2:
        raise ValueError(f"{where}: sweep keys inside {section}, e.g. {section}.<name>")
    if section in ("model", "baseline") and parts[1:] == ["name"]:
        raise ValueError(
            f"{where}: the architecture is fixed; compare architectures in "
            f"separate sweeps ({PROTOCOL})"
        )
    if section in ("model", "baseline") and parts[1] == "train":
        setting = parts[2] if len(parts) == 3 else None
        if setting == "seed":
            raise ValueError(f"{where}: the seed is not tuned; list seeds in sweep.seeds")
        if setting not in cfg["train"]:
            raise ValueError(
                f"{where}: not a training setting; valid: {sorted(cfg['train'].keys() - {'seed'})}"
            )
        return
    node: Any = cfg["graph"]
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"{where}: no such key in config/graphs/{cfg['graph']['name']}.yaml")
        node = node[part]


def expand(grid: dict[str, list]) -> list[Combo]:
    """Every combination of the grid's values (Cartesian product), in grid order."""
    keys = list(grid)
    combos = []
    for index, picks in enumerate(itertools.product(*(range(len(grid[k])) for k in keys))):
        values = {k: grid[k][i] for k, i in zip(keys, picks, strict=True)}
        name = ", ".join(
            f"{k}={v}" if not isinstance(v, list | dict) else f"{k}=#{i + 1}"
            for (k, v), i in zip(values.items(), picks, strict=True)
        )
        combos.append(Combo(index, values, name))
    return combos


def apply(cfg: dict, overrides: dict[str, Any]) -> dict:
    """A copy of `cfg` with each graph-file path set to its value."""
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        *parents, last = key.split(".")
        node = out["graph"]
        for part in parents:
            if node.get(part) is None:  # e.g. a model without its own train: block yet
                node[part] = {}
            node = node[part]
            if not isinstance(node, dict):
                raise ValueError(f"{key}: {part} is not a mapping")
        node[last] = copy.deepcopy(value)
    return out


def required_history(cfg: dict) -> int:
    """The `min_history_bars` this config needs: the file's value, the longest feature
    window, or the builder's price window (in returns), whichever is largest. The builder
    is built on empty inputs to read its lookbacks, which also checks its params."""
    graph = cfg["graph"]
    if graph["builder"] not in BUILDERS:
        raise ValueError(f"unknown builder {graph['builder']!r}; valid: {sorted(BUILDERS)}")
    empty = pl.DataFrame(schema={"timestamp": pl.Datetime("us", "UTC")})
    names = ["prices", "features", *graph.get("sources", {})]
    builder = BUILDERS[graph["builder"]](
        {n: empty for n in names},
        empty["timestamp"],
        **graph.get("params", {}),
    )
    price = builder.lookbacks().get("prices")
    features = graph["features"]
    longest = max(s["bars"] for s in features["list"])
    return max(features["min_history_bars"], longest, price - 1 if price else 0)


def check_combo(cfg: dict, grid: str) -> None:
    """Raise before any training if this combination can't run."""
    graph = cfg["graph"]
    BUILDERS[graph["builder"]].check_config(cfg)
    required_history(cfg)
    if unknown := sorted({s["name"] for s in graph["features"]["list"]} - FEATURES.keys()):
        raise ValueError(f"unknown features {unknown}; valid: {sorted(FEATURES)}")
    labels = [graph[role]["name"] for role in run.ROLES]
    if len(set(labels)) < len(labels):
        raise ValueError(f"model and baseline need different names, got {labels}")
    nodes, edges = model_columns(graph)
    sample = Data(x=torch.zeros(1, len(nodes)), edge_attr=torch.zeros(0, len(edges)))
    for role in run.ROLES:
        settings = run.train_settings(cfg, role)
        if settings["loss"] not in LOSSES:
            raise ValueError(f"unknown loss {settings['loss']!r}; valid: {sorted(LOSSES)}")
        try:
            build_model(graph[role], sample)
        except TypeError as e:
            raise ValueError(f"sweep.{grid}: {role} {graph[role]['name']}: {e}") from e


def with_seed(cfg: dict, seed: int) -> dict:
    return cfg | {"train": cfg["train"] | {"seed": seed}}


def _stage(
    stage: int,
    combos: list[Combo],
    base: dict,
    seeds: list[int],
    device: str,
    parent_id: str,
    rows: list[dict],
    upstream: Best | None,
) -> Best:
    """Train the stage's role on every combination x seed; return the best combination.

    Stage 1 builds once per distinct data-side setting; stage 2 reuses `upstream.built`
    (its combinations never change the data)."""
    role = "model" if stage == 1 else "baseline"
    groups: dict[str, list[Combo]] = {}
    for c in combos:
        shared = {k: v for k, v in c.overrides.items() if k.split(".")[0] in SHARED}
        groups.setdefault(json.dumps(shared, sort_keys=True), []).append(c)
    best: Best | None = None
    done = 0
    for key, members in groups.items():
        if upstream is None:
            built = run.build(apply(base, json.loads(key)))
        else:
            built = upstream.built
        if done == 0:
            log.info("stage %d (%s): %d search folds", stage, role, len(built.folds))
        for c in members:
            cfg_c = apply(base, c.overrides)
            results, scores = {}, []
            for seed in seeds:
                cfg_s = with_seed(cfg_c, seed)
                r = run.train(built, role, cfg_s, device)
                summary = run.summarize({r.label: r}, cfg_s, test=False)
                run_id = run.log_run(
                    cfg_s,
                    summary,
                    {r.label: r},
                    built.manifests,
                    predictions=False,
                    parent_run_id=parent_id,
                    tags={
                        "sweep.stage": str(stage),
                        "sweep.combo": str(c.index),
                        "sweep.seed": str(seed),
                        "sweep.overrides": json.dumps(c.overrides),
                    },
                    run_name=f"stage {stage} #{c.index} seed {seed}",
                )
                fold_val = summary[r.label]["fold_val_ic"]
                rows.extend(
                    {
                        "stage": stage,
                        "combo": c.index,
                        "name": c.name,
                        "seed": seed,
                        "fold": fold,
                        "label": r.label,
                        "val_ic": ic,
                        "epochs": len(h),
                        "run_id": run_id,
                    }
                    for fold, (ic, h) in enumerate(zip(fold_val, r.histories, strict=True), 1)
                )
                results[seed] = r
                scores.append(summary[r.label]["val_ic"])
            score = statistics.fmean(scores)
            done += 1
            log.info(
                "stage %d %d/%d [%s]: mean val IC %.4f over %d seeds",
                stage,
                done,
                len(combos),
                c.name,
                score,
                len(seeds),
            )
            if better(score, c.index, best):
                best = Best(c, cfg_c, score, results, built)
    assert best is not None
    return best


def better(score: float, index: int, best: Best | None) -> bool:
    """Whether a combination beats the best so far: higher mean validation IC, or equal
    and earlier in grid order (stage 1 runs combinations grouped by build, not in order)."""
    return best is None or (score, -index) > (best.score, -best.combo.index)


def _final_runs(
    cfg: dict, best1: Best, best2: Best, seeds: list[int], device: str, parent_id: str
) -> tuple[list[str], dict[int, dict]]:
    """One run per seed with both winners: test predictions, test metrics, importance."""
    ids, tests = [], {}
    for seed in seeds:
        cfg_s = with_seed(cfg, seed)
        results = {r.label: r for r in (best1.results[seed], best2.results[seed])}
        for r in results.values():
            r.importances = run.importance(r, best2.built, cfg_s, device)
        summary = run.summarize(results, cfg_s)
        ids.append(
            run.log_run(
                cfg_s,
                summary,
                results,
                best2.built.manifests,
                parent_run_id=parent_id,
                tags={"sweep.stage": "final", "sweep.seed": str(seed)},
                run_name=f"final seed {seed}",
            )
        )
        tests[seed] = {label: s["mean_ic"] for label, s in summary.items()}
    return ids, tests


def _summary(
    table: pl.DataFrame,
    best1: Best,
    best2: Best,
    seeds: list[int],
    tests: dict[int, dict],
    finals: list[str],
) -> dict:
    """Winners, their spread over seeds, how often each wins fold by fold, and the lift."""
    out: dict = {}
    for stage, best in ((1, best1), (2, best2)):
        rows = table.filter(pl.col("stage") == stage)
        by_seed = (
            rows.filter(pl.col("combo") == best.combo.index)
            .group_by("seed")
            .agg(pl.col("val_ic").mean())
            .sort("seed")
        )
        # Per fold: the combination with the best mean (over seeds) validation IC.
        per_fold = rows.group_by("fold", "combo").agg(pl.col("val_ic").mean())
        fold_best = (
            per_fold.sort("val_ic", "combo", descending=[True, False])
            .group_by("fold", maintain_order=True)
            .first()
        )
        wins = int((fold_best["combo"] == best.combo.index).sum())
        out[f"stage_{stage}"] = {
            "role": "model" if stage == 1 else "baseline",
            "label": rows["label"][0],
            "combinations": rows["combo"].n_unique(),
            "winner": best.combo.index,
            "winner_name": best.combo.name,
            "overrides": best.combo.overrides,
            "val_ic": best.score,
            "val_ic_sd_over_seeds": statistics.stdev(by_seed["val_ic"]) if len(seeds) > 1 else 0.0,
            "val_ic_by_seed": dict(zip(by_seed["seed"], by_seed["val_ic"], strict=True)),
            "folds_won": wins,
            "folds": fold_best.height,
        }
    gnn, base = out["stage_1"]["label"], out["stage_2"]["label"]
    val = {
        stage: table.filter((pl.col("stage") == stage) & (pl.col("combo") == best.combo.index))
        for stage, best in ((1, best1), (2, best2))
    }
    lift = (
        val[1]
        .join(val[2], on=["seed", "fold"], suffix="_baseline")
        .select("seed", "fold", lift=pl.col("val_ic") - pl.col("val_ic_baseline"))
    )
    out["val_lift"] = statistics.fmean(lift["lift"])
    out["val_lift_by_fold_and_seed"] = lift.sort("seed", "fold").rows(named=True)
    out["test_ic_by_seed"] = tests
    out["test_lift"] = statistics.fmean(t[gnn] - t[base] for t in tests.values())
    out["final_runs"] = finals
    return out


def _log_parent(
    client: mlflow.MlflowClient,
    parent_id: str,
    cfg: dict,
    spec: dict,
    table: pl.DataFrame,
    summary: dict,
    frozen: dict,
    built: run.Built,
) -> None:
    """Config, git SHA, manifests, spec, results table, summary and `frozen.yaml`."""
    flat = run._flatten({k: v for k, v in cfg.items() if k not in ("data_root", "graph")})
    flat |= run._flatten({k: v for k, v in cfg["graph"].items() if k != "sweep"}, "graph.")
    params = [Param(k, v) for k, v in flat.items()]
    for i in range(0, len(params), 100):  # log_batch takes at most 100 params per call
        client.log_batch(parent_id, params=params[i : i + 100])
    client.set_tag(parent_id, "git_sha", git_sha())
    for stage in ("stage_1", "stage_2"):
        s = summary[stage]
        client.log_metric(parent_id, f"{s['label']}_val_ic", s["val_ic"])
        client.log_metric(parent_id, f"{s['label']}_folds_won", s["folds_won"])
    client.log_metric(parent_id, "val_lift", summary["val_lift"])
    client.log_metric(parent_id, "test_lift", summary["test_lift"])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (out / "sweep.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        (out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
        (out / "frozen.yaml").write_text(frozen_yaml(frozen, cfg["graph"]["name"], parent_id))
        table.write_csv(out / "results.csv")
        for dataset, manifest in built.manifests.items():
            (out / f"manifest_{dataset}.json").write_text(json.dumps(manifest, indent=2))
        client.log_artifacts(parent_id, str(out))


def frozen_yaml(frozen: dict, graph_name: str, parent_id: str) -> str:
    """The tuned values as YAML to paste into the graph file."""
    head = f"# gnns sweep {parent_id}: tuned values for config/graphs/{graph_name}.yaml\n"
    return head + yaml.safe_dump(frozen, sort_keys=False)


def _nest(flat: dict[str, Any]) -> dict:
    """{"model.train.lr": 0.01} -> {"model": {"train": {"lr": 0.01}}}."""
    out: dict = {}
    for key, value in flat.items():
        *parents, last = key.split(".")
        node = out
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = value
    return out
