"""Universe definitions: one `config/universes/<name>.yaml` per universe.

A universe file has a `kind` (docs/CONTRACTS.md, "Storage layout"):
- `list`: fixed `tickers`; every ticker is a node on every date it has a bar.
- `membership`: members change over time ("Membership universes"); the file names the
  constituent list and the trailing windows of its `adv_<w>` columns (`adv_windows`).
"""

from pathlib import Path

import yaml

from gnns.config import CONFIG_DIR

KINDS = ("list", "membership")


def spec(name: str, config_dir: Path = CONFIG_DIR) -> dict:
    """The validated definition of universe `name`."""
    path = config_dir / "universes" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"universe {name!r}: no definition at {path}")
    s = yaml.safe_load(path.read_text()) or {}
    if s.get("kind") not in KINDS:
        raise ValueError(f"{path}: kind must be one of {list(KINDS)}, got {s.get('kind')!r}")
    if s["kind"] == "list":
        tickers = s.get("tickers")
        if not tickers or not all(isinstance(t, str) and t for t in tickers):
            raise ValueError(f"{path}: a list universe needs a non-empty list of tickers")
        if dupes := sorted({t for t in tickers if tickers.count(t) > 1}):
            raise ValueError(f"{path}: duplicate tickers {dupes}")
    else:
        w = s.get("adv_windows", [20, 63])
        if not w or not all(isinstance(x, int) and x > 0 for x in w) or len(set(w)) < len(w):
            raise ValueError(f"{path}: adv_windows must be distinct positive integers, got {w!r}")
    return s
