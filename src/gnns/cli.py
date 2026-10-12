"""The `gnns` command. Each pipeline stage is a subcommand."""

import argparse

from gnns.config import load_config
from gnns.data.build import build


def main() -> None:
    parser = argparse.ArgumentParser(prog="gnns")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-data", help="fetch, adjust, validate and write the configured datasets")
    args = parser.parse_args()

    cfg = load_config()
    if args.command == "build-data":
        for dest in build(cfg):
            print(f"wrote {dest}")
