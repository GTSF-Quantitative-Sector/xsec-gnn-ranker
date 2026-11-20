"""Build the configured datasets: fetch -> adjust -> validate -> write.

Two commands, each all-or-nothing (everything validates before the first write):
- `gnns build-universe` (`build_universe`): each membership universe in `data.universes`
  (`gnns.data.universe`) from the cached WRDS extracts (`gnns fetch-wrds`) and Massive's
  unadjusted bars: its universe table, raw bars and delistings, and the global security
  master, with review reports in `<data_root>/<universe>/reports/universe/`.
- `gnns build-data` (`build`): every universe's prices, plus the selected graph's buildable
  sources. A list universe is fetched from Massive; a membership universe's prices are its
  raw bars × CRSP's factors (`gnns.data.adjustment`), with reports in
  `<data_root>/<universe>/reports/prices/`. A construction's extra datasets are the
  selected graph's `sources:` entries that name a `kind` in `SOURCES`; each kind's function
  fetches and validates its dataset and returns it unwritten.
"""

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from gnns.config import CONFIG_DIR, ROOT, git_sha
from gnns.data import adjustment, macro, massive, store, universe, universes

log = logging.getLogger(__name__)

# kind -> fetch(spec, start, end) -> (data, manifest fields beyond built_at/git_sha/start/end)
SOURCES: dict[str, Callable[[dict, date, date], tuple[pl.DataFrame, dict]]] = {
    "macro": macro.fetch_macro,
}
Pending = list[tuple[Path, pl.DataFrame, dict, str]]  # (dest, data, manifest, file)


def build(cfg: dict) -> list[Path]:
    """Build every universe's prices in `data.universes` on the shared bar size and date
    range, plus the selected graph's buildable sources (`buildable_sources`).

    Everything is fetched and validated before anything is written, so a failure in one
    dataset leaves every existing dataset untouched.
    """
    bar, start, end = _window(cfg)
    specs = buildable_sources(cfg)  # before any download, so a bad kind costs nothing
    root = ROOT / cfg["data_root"]
    stamp = {"built_at": datetime.now(UTC).isoformat(), "git_sha": git_sha()}
    window = {"bar": bar, "start": str(start), "end": str(end)}
    pending: Pending = []
    reports: dict[Path, dict[str, pl.DataFrame]] = {}
    for name in cfg["data"]["universes"]:
        spec = universes.spec(name, CONFIG_DIR)
        if spec["kind"] == "list":
            tickers, prices = _fetch(spec["tickers"], start, end)
            manifest = {"source": "massive", "universe": name, "tickers": tickers}
            pending.append((root / name / bar, prices, manifest | window | stamp, "prices.parquet"))
            continue
        b = adjustment.build(name, start, end, root)
        for line in b.summary:
            log.info("%s: %s", name, line)
        manifest = b.manifest | {"universe": name} | window | stamp
        pending.append((root / name / bar, b.prices, manifest, "prices.parquet"))
        reports[root / name / "reports" / "prices"] = b.reports
    for name, spec in specs.items():
        df, extras = SOURCES[spec["kind"]](spec, start, end)
        pending.append((root / spec["path"], df, extras | window | stamp, f"{name}.parquet"))
    return _write(pending, reports)


def build_universe(cfg: dict) -> list[Path]:
    """Build each membership universe in `data.universes`: its universe table, raw bars
    and delistings, and the security master. Raises if none is configured."""
    _, start, end = _window(cfg)
    root = ROOT / cfg["data_root"]
    names = [
        n for n in cfg["data"]["universes"] if universes.spec(n, CONFIG_DIR)["kind"] == "membership"
    ]
    if not names:
        raise ValueError("data.universes has no membership universe to build")
    if len(names) > 1:
        raise NotImplementedError(
            f"membership universes {names}: one at a time so far (they share one security "
            "master; see docs/CONTRACTS.md open questions)"
        )
    stamp = {"built_at": datetime.now(UTC).isoformat(), "git_sha": git_sha()}
    pending: Pending = []
    reports: dict[Path, dict[str, pl.DataFrame]] = {}
    for name in names:
        b = universe.build(name, universes.spec(name, CONFIG_DIR), start, end, root, CONFIG_DIR)
        for line in b.summary:
            log.info("%s: %s", name, line)
        m = b.manifests
        pending += [
            (root / name / "universe", b.universe, m["universe"] | stamp, "universe.parquet"),
            (root / name / "raw_bars", b.raw_bars, m["raw_bars"] | stamp, "raw_bars.parquet"),
            (
                root / name / "delistings",
                b.delistings,
                m["delistings"] | stamp,
                "delistings.parquet",
            ),
            (
                root / "security_master",
                b.security_master,
                m["security_master"] | stamp,
                "security_master.parquet",
            ),
        ]
        reports[root / name / "reports" / "universe"] = b.reports
    return _write(pending, reports)


def _window(cfg: dict) -> tuple[str, date, date]:
    bar, start, end = cfg["data"]["bar"], cfg["data"]["start"], cfg["data"]["end"]
    if bar != "1d":
        raise NotImplementedError(f"bar {bar!r}: only 1d so far (see CONTRACTS.md open questions)")
    return bar, start, end


def _write(pending: Pending, reports: dict[Path, dict[str, pl.DataFrame]]) -> list[Path]:
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
