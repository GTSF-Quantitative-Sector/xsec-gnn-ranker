"""Load config/config.yaml merged with the gitignored config/machine.yaml."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"


def load_config() -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / "config.yaml").read_text()) or {}
    machine_path = CONFIG_DIR / "machine.yaml"
    machine = (yaml.safe_load(machine_path.read_text()) or {}) if machine_path.exists() else {}
    return {**cfg, "data_root": "data", **machine}
