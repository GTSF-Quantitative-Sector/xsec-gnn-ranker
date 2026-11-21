"""Evidence for an identifier override: what Datastream and Massive say about one security.

    python scripts/override_evidence.py 39979 --symbols JAVA --start 2010-01-01 --end 2010-02-15

Prints the security's Datastream history (names, tickers, CUSIPs, S&P list spells) from the
`gnns fetch-wrds` cache, its rows in the built universe table (membership and the Massive
ticker the build chose), and, for each candidate Massive symbol, its unadjusted bars in the
window and whether they continue the security's own bars at both ends. Reads no CRSP
values (WRDS AI rules: CRSP stays aggregate-only in an agent's context).
"""

import argparse
import math
from datetime import date, timedelta

import polars as pl

from gnns.config import ROOT, load_config
from gnns.data import massive, store, universe, wrds

PAD = timedelta(days=10)
MAX_SEAM = universe.MAX_SEAM


def _ds(root, name: str, sid: str) -> pl.DataFrame:
    df, _ = wrds.load(root, name)
    return df.filter(pl.col("infocode").cast(pl.Int64).cast(pl.String) == sid)


def _print(title: str, df: pl.DataFrame) -> None:
    print(f"\n{title}:")
    if df.is_empty():
        print("  (none)")
        return
    with pl.Config(tbl_rows=60, tbl_width_chars=160, fmt_str_lengths=60):
        print(df)


def _move(a: float | None, b: float | None) -> str:
    if a is None or b is None or a <= 0 or b <= 0:
        return "n/a"
    m = math.log(b / a)
    return f"{m:+.3f} ({'continues' if abs(m) <= MAX_SEAM else 'BREAK'})"


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("security_id")
    ap.add_argument("--symbols", nargs="*", default=[], help="candidate Massive tickers")
    ap.add_argument("--start", type=date.fromisoformat, help="window start (default: data.start)")
    ap.add_argument("--end", type=date.fromisoformat, help="window end (default: data.end)")
    args = ap.parse_args()
    cfg = load_config()
    root = ROOT / cfg["data_root"]
    sid = args.security_id
    lo, hi = args.start or cfg["data"]["start"], args.end or cfg["data"]["end"]

    print(f"security {sid}, window {lo} .. {hi}")
    _print(
        "Datastream names",
        _ds(root, "ds_names", sid).select(
            "startdate", "enddate", "dsqtname", "dscmpycode", "isin", "delistdate"
        ),
    )
    _print(
        "Datastream tickers", _ds(root, "ds_tickers", sid).select("startdate", "enddate", "ticker")
    )
    _print("Datastream CUSIPs", _ds(root, "ds_cusips", sid).select("startdate", "enddate", "cusip"))
    _print("S&P list spells", _ds(root, "ds_constituents", sid).select("startdate", "enddate"))

    # The built table (identifiers from Datastream and Massive only; no CRSP columns).
    own = pl.DataFrame(schema={"date": pl.Date, "close": pl.Float64})
    sessions = pl.Series("date", [], pl.Date)
    if universe.exists(root, "sp500"):
        t, _ = universe.load(root, "sp500")
        t = t.filter(pl.col("security_id") == sid).sort("timestamp")
        runs = (
            t.with_columns(date=pl.col("timestamp").dt.date())
            .with_columns(
                run=(
                    pl.col("ticker_massive").ne_missing(pl.col("ticker_massive").shift(1))
                ).cum_sum()
            )
            .group_by("run", "ticker_massive")
            .agg(
                first=pl.col("date").min(),
                last=pl.col("date").max(),
                rows=pl.len(),
                member=pl.col("member").sum(),
            )
            .sort("first")
            .drop("run")
        )
        _print("Universe table: Massive ticker runs (rows, member sessions)", runs)
        raw_dest = root / "sp500" / "raw_bars"
        if (raw_dest / "manifest.json").exists():
            lf, _ = store.open(raw_dest)
            raw = lf.with_columns(date=pl.col("timestamp").dt.date()).collect()
            sessions = raw["date"].unique().sort()
            own = raw.filter(pl.col("security_id") == sid).select("date", "close").sort("date")
    else:
        print("\n(no universe table yet: run gnns build-universe for membership and runs)")

    before = own.filter(pl.col("date") < lo).tail(1)
    after = own.filter(pl.col("date") > hi).head(1)
    window_sessions = sessions.filter(sessions.is_between(lo, hi))
    for sym in args.symbols:
        bars = massive.daily_bars(sym, lo - PAD, hi + PAD, adjusted=False).with_columns(
            date=pl.col("timestamp").dt.date()
        )
        inside = bars.filter(pl.col("date").is_between(lo, hi)).sort("date")
        print(f"\n{sym}: {inside.height} bars in the window", end="")
        if inside.is_empty():
            print()
            continue
        print(
            f", {inside['date'].min()} .. {inside['date'].max()}, "
            f"close {inside['close'][0]:.4g} .. {inside['close'][-1]:.4g}"
        )
        if len(window_sessions):
            missing = window_sessions.filter(~window_sessions.is_in(inside["date"].implode()))
            print(
                f"  sessions in the window without a {sym} bar: {len(missing)}"
                + (f" (first {missing[0]}, last {missing[-1]})" if len(missing) else "")
            )
        b = before["close"][0] if before.height else None
        a = after["close"][0] if after.height else None
        b_day = before["date"][0] if before.height else "-"
        a_day = after["date"][0] if after.height else "-"
        print(f"  from the security's last bar before ({b_day}): {_move(b, inside['close'][0])}")
        print(f"  to its first bar after ({a_day}): {_move(inside['close'][-1], a)}")


if __name__ == "__main__":
    main()
