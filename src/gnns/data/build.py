"""Build the configured price datasets: fetch -> adjust -> validate -> write."""

import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from gnns.config import CONFIG_DIR, ROOT
from gnns.data import massive, store


def build(cfg: dict) -> list[Path]:
    """Build every universe in `data.universes` on the shared bar size and date range.

    All universes are fetched and validated before any is written, so a failure in one
    leaves every existing dataset untouched.
    """
    bar, start, end = cfg["data"]["bar"], cfg["data"]["start"], cfg["data"]["end"]
    if bar != "1d":
        raise NotImplementedError(f"bar {bar!r}: only 1d so far (see CONTRACTS.md open questions)")

    built_at = datetime.now(UTC).isoformat()
    fetched = {u: _fetch(u, start, end) for u in cfg["data"]["universes"]}

    git_sha, dests = _git_sha(), []
    for universe, (tickers, prices) in fetched.items():
        dest = ROOT / cfg["data_root"] / universe / bar
        manifest = {
            "universe": universe,
            "bar": bar,
            "start": str(start),
            "end": str(end),
            "tickers": tickers,
            "built_at": built_at,
            "git_sha": git_sha,
        }
        store.write(prices, dest, manifest)
        dests.append(dest)
    return dests


def _fetch(universe: str, start: date, end: date) -> tuple[list[str], pl.DataFrame]:
    tickers = (CONFIG_DIR / "universes" / f"{universe}.txt").read_text().split()
    prices = pl.concat([massive.daily_bars(t, start, end) for t in tickers])
    divs = pl.concat([massive.dividends(t) for t in tickers])
    prices = store.adjust(prices, divs)
    store.validate(prices, tickers, start, end)
    return tickers, prices


def _git_sha() -> str:
    """HEAD, suffixed -dirty if src/ has uncommitted changes (config edits don't count)."""

    def git(*args: str) -> str:
        cmd = ["git", *args]
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout

    dirty = git("status", "--porcelain", "--", "src").strip()
    return git("rev-parse", "HEAD").strip() + ("-dirty" if dirty else "")
