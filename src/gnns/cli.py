"""The `gnns` command. Each pipeline stage is a subcommand."""

import argparse
import logging

from gnns.config import load_config
from gnns.data.build import build
from gnns.graphs.build import build_snapshots


def main() -> None:
    parser = argparse.ArgumentParser(prog="gnns")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-data", help="fetch, adjust, validate and write the configured datasets")
    sub.add_parser("build-graphs", help="build the configured graph's snapshots and summarize them")
    sub.add_parser("train", help="train model + baseline on search folds; log metrics to MLflow")
    rep = sub.add_parser("report", help="make evaluation charts for one MLflow run")
    rep.add_argument("run_id", help="MLflow run id, as printed by gnns train or shown in mlflow ui")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config()
    if args.command == "build-data":
        for dest in build(cfg):
            print(f"wrote {dest}")
    elif args.command == "build-graphs":
        snaps = build_snapshots(cfg)
        nodes = [len(s.node_ids) for s in snaps]
        edges = [s.edge_index.shape[1] for s in snaps]
        print(f"{cfg['graph']['name']}: {len(snaps)} snapshots")
        print(f"  {snaps[0].timestamp} to {snaps[-1].timestamp}")
        print(f"  nodes {min(nodes)}-{max(nodes)}, edges {min(edges)}-{max(edges)}")
    elif args.command == "train":
        from gnns.train.run import run  # torch and mlflow load only for this command

        run_id = run(cfg)
        print(f"MLflow run {run_id}")
        print("  browse runs: mlflow ui --backend-store-uri sqlite:///mlflow.db")
        print(f"  make charts: gnns report {run_id}")
    elif args.command == "report":
        from gnns.eval.report import report

        for name in report(args.run_id, cfg["data_root"]):
            print(f"logged {name} to run {args.run_id}")
