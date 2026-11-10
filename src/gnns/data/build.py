"""Build the configured datasets: fetch -> adjust -> validate -> write.

Shared datasets are the price universes in `data.universes`. A construction's extra
datasets are the selected graph's `sources:` entries that name a `kind` in `SOURCES`; each
kind's function fetches and validates its dataset and returns it unwritten.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from gnns.config import CONFIG_DIR, ROOT, git_sha
from gnns.data import macro, massive, store

# kind -> fetch(spec, start, end) -> (data, manifest fields beyond built_at/git_sha/start/end)
SOURCES: dict[str, Callable[[dict, date, date], tuple[pl.DataFrame, dict]]] = {
    "macro": macro.fetch_macro,
}


def build(cfg: dict) -> list[Path]:
    """Build every universe in `data.universes` on the shared bar size and date range,
    plus the selected graph's buildable sources (`buildable_sources`).

    Everything is fetched and validated before anything is written, so a failure in one
    dataset leaves every existing dataset untouched.
    """
    bar, start, end = cfg["data"]["bar"], cfg["data"]["start"], cfg["data"]["end"]
    if bar != "1d":
        raise NotImplementedError(f"bar {bar!r}: only 1d so far (see CONTRACTS.md open questions)")
    specs = buildable_sources(cfg)  # before any download, so a bad kind costs nothing

    built_at = datetime.now(UTC).isoformat()
    fetched = {u: _fetch(u, start, end) for u in cfg["data"]["universes"]}
    sources = {name: SOURCES[spec["kind"]](spec, start, end) for name, spec in specs.items()}

    sha, dests = git_sha(), []
    for universe, (tickers, prices) in fetched.items():
        dest = ROOT / cfg["data_root"] / universe / bar
        manifest = {
            "source": "massive",
            "universe": universe,
            "bar": bar,
            "start": str(start),
            "end": str(end),
            "tickers": tickers,
            "built_at": built_at,
            "git_sha": sha,
        }
        store.write(prices, dest, manifest)
        dests.append(dest)
    for name, (df, extras) in sources.items():
        dest = ROOT / cfg["data_root"] / specs[name]["path"]
        manifest = extras | {
            "bar": bar,
            "start": str(start),
            "end": str(end),
            "built_at": built_at,
            "git_sha": sha,
        }
        store.write(df, dest, manifest, file=f"{name}.parquet")
        dests.append(dest)
    return dests


def buildable_sources(cfg: dict) -> dict[str, dict]:
    """The selected graph's `sources:` entries with a `kind`; raises on an unknown kind.

    Entries without a `kind` are produced outside `gnns build-data`. Other graphs'
    sources are never built: select a graph (`graph:` in machine.yaml) to build its own.
    """
    graph = cfg.get("graph") or {}
    specs = {n: s for n, s in graph.get("sources", {}).items() if "kind" in s}
    if unknown := sorted({s["kind"] for s in specs.values()} - SOURCES.keys()):
        raise ValueError(
            f"graph {graph.get('name')!r}: unknown source kind {unknown}; valid: {sorted(SOURCES)}"
        )
    return specs


def _fetch(universe: str, start: date, end: date) -> tuple[list[str], pl.DataFrame]:
    tickers = (CONFIG_DIR / "universes" / f"{universe}.txt").read_text().split()
    prices = pl.concat([massive.daily_bars(t, start, end) for t in tickers])
    divs = pl.concat([massive.dividends(t) for t in tickers])
    prices = store.adjust(prices, divs)
    store.validate(prices, tickers, start, end)
    return tickers, prices
