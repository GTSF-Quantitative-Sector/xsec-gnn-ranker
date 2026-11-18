"""Build the configured datasets: fetch -> adjust -> validate -> write.

Shared datasets are the universes in `data.universes`: a list universe is one price
dataset; a membership universe (`gnns.data.equities`) is its prices, membership and
delistings plus the global security master, built from the cached WRDS extracts
(`gnns fetch-wrds`), with review reports in `<data_root>/<universe>/reports/`. A
construction's extra datasets are the selected graph's `sources:` entries that name a
`kind` in `SOURCES`; each kind's function fetches and validates its dataset and returns
it unwritten.
"""

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from gnns.config import CONFIG_DIR, ROOT, git_sha
from gnns.data import equities, macro, massive, store, universes

log = logging.getLogger(__name__)

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

    built_at, sha = datetime.now(UTC).isoformat(), git_sha()
    root = ROOT / cfg["data_root"]
    stamp = {"built_at": built_at, "git_sha": sha}
    window = {"bar": bar, "start": str(start), "end": str(end)}
    # (dest, data, manifest, file), all validated before the first write.
    pending: list[tuple[Path, pl.DataFrame, dict, str]] = []
    reports: dict[Path, dict[str, pl.DataFrame]] = {}
    for universe in cfg["data"]["universes"]:
        spec = universes.spec(universe, CONFIG_DIR)
        if spec["kind"] == "list":
            tickers, prices = _fetch(spec["tickers"], start, end)
            manifest = {"source": "massive", "universe": universe, "tickers": tickers}
            pending.append(
                (root / universe / bar, prices, manifest | window | stamp, "prices.parquet")
            )
            continue
        b = equities.build(universe, spec, start, end, root, CONFIG_DIR)
        for line in b.summary:
            log.info("%s: %s", universe, line)
        m = b.manifests
        pending += [
            (
                root / universe / bar,
                b.prices,
                m["prices"] | {"universe": universe} | window | stamp,
                "prices.parquet",
            ),
            (
                root / universe / "membership",
                b.membership,
                m["membership"] | stamp,
                "membership.parquet",
            ),
            (
                root / universe / "delistings",
                b.delistings,
                m["delistings"] | {"universe": universe} | stamp,
                "delistings.parquet",
            ),
            (
                root / "security_master",
                b.security_master,
                m["security_master"] | stamp,
                "security_master.parquet",
            ),
        ]
        reports[root / universe / "reports"] = b.reports
    for name, spec in specs.items():
        df, extras = SOURCES[spec["kind"]](spec, start, end)
        pending.append((root / spec["path"], df, extras | window | stamp, f"{name}.parquet"))

    dests = []
    for dest, df, manifest, file in pending:
        store.write(df, dest, manifest, file=file)
        dests.append(dest)
    for folder, named in reports.items():
        _write_reports(folder, named)
    return dests


def _write_reports(folder: Path, named: dict[str, pl.DataFrame]) -> None:
    """Review reports as CSV files, replacing the previous build's (an empty report is
    removed, so the folder lists only what needs a look)."""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.csv"):
        old.unlink()
    for name, df in named.items():
        if df.height:
            df.write_csv(folder / f"{name}.csv")
            log.info("review report %s: %d rows", folder / f"{name}.csv", df.height)


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


def _fetch(tickers: list[str], start: date, end: date) -> tuple[list[str], pl.DataFrame]:
    """A list universe's prices: Massive split-adjusted bars × dividend factors."""
    prices = pl.concat([massive.daily_bars(t, start, end) for t in tickers])
    divs = pl.concat([massive.dividends(t) for t in tickers])
    prices = store.adjust(prices, divs)
    store.validate(prices, tickers, start, end)
    return tickers, prices
