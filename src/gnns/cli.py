"""The `gnns` command. Each pipeline stage is a subcommand."""

import argparse

from gnns.config import load_config
from gnns.data.build import build
from gnns.graphs.build import build_snapshots


def main() -> None:
    parser = argparse.ArgumentParser(prog="gnns")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-data", help="fetch, adjust, validate and write the configured datasets")
    sub.add_parser("build-graphs", help="build the configured graph's snapshots and summarize them")
    args = parser.parse_args()

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
