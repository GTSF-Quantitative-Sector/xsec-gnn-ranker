"""WRDS: the connection, and the cached extracts the sp500 universe is built from.

`gnns fetch-wrds` is the only command that connects (each connection may wait for a Duo
approval). It writes every extract as a dataset under `<data_root>/wrds/<name>/`
(docs/CONTRACTS.md, "Storage layout"); `gnns build-universe` and `gnns build-data` only
read them. Credentials come
from `.env` (`WRDS_USERNAME`, `WRDS_PASSWORD`; README, "WRDS access").

Extracts are raw source rows, never modified here. Dates stay as the source writes them
(Datastream's open end date 2079-06-05 included); the build interprets them.
"""

import logging
import os
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
import psycopg2
from dotenv import load_dotenv

from gnns.config import CONFIG_DIR, ROOT, git_sha
from gnns.data import store, universes

log = logging.getLogger(__name__)
HOST, PORT, DBNAME = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"
TIMEOUT_MS = 900_000  # per query
DS = "tr_ds_equities"


def connect() -> "psycopg2.extensions.connection":
    """An autocommit connection, so one failed query never aborts the next.

    Use it with try/finally `conn.close()`, not `with conn:`: since psycopg2 2.9 the
    context manager opens a transaction even in autocommit mode.
    """
    load_dotenv(ROOT / ".env")
    user = os.environ.get("WRDS_USERNAME")
    if not user:
        raise RuntimeError("WRDS_USERNAME is not set; add it to .env (README, 'WRDS access').")
    # None is dropped by psycopg2, and libpq then falls back to the pgpass file.
    password = os.environ.get("WRDS_PASSWORD") or None
    log.info("connecting to WRDS as %s (approve the Duo push if your phone asks)", user)
    try:
        conn = psycopg2.connect(
            host=HOST,
            port=PORT,
            dbname=DBNAME,
            user=user,
            password=password,
            sslmode="require",
            connect_timeout=60,
        )
    except psycopg2.OperationalError as e:
        raise RuntimeError(
            f"could not connect to WRDS: {e}\nCheck WRDS_USERNAME and WRDS_PASSWORD in .env."
        ) from e
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {TIMEOUT_MS}")
    return conn


def query(cur, sql: str, params: dict | None = None) -> pl.DataFrame:
    """Run `sql` and return its rows as a frame; NUMERIC columns become Float64."""
    cur.execute(sql, params)
    names = [d.name for d in cur.description]
    rows = cur.fetchall()
    decimal_cols = {
        n for i, n in enumerate(names) if any(isinstance(r[i], Decimal) for r in rows[:1000])
    }
    if decimal_cols:
        rows = [tuple(float(v) if isinstance(v, Decimal) else v for v in r) for r in rows]
    df = pl.DataFrame(rows, schema=names, orient="row", infer_schema_length=None)
    return df.with_columns(pl.col(n).cast(pl.Float64) for n in decimal_cols)


# --- Extracts ------------------------------------------------------------------------------
#
# Each extract is (name, table, function(cur, ctx) -> frame). They run in order; `ctx`
# carries what later extracts filter on (the list's infocodes, their CUSIPs, the PERMNOs).


def _constituents(cur, ctx: dict) -> pl.DataFrame:
    df = query(
        cur,
        f"SELECT indexlistintcode, infocode, startdate, enddate FROM {DS}.ds2constmth "
        "WHERE indexlistintcode = %(code)s",
        {"code": ctx["list_code"]},
    )
    # Spells ending in the 40 days before `start` still matter: the last list before
    # `start` decides membership on its first sessions.
    in_range = df.filter(
        pl.col("enddate") >= ctx["start"] - timedelta(days=40), pl.col("infocode").is_not_null()
    )
    ctx["infocodes"] = sorted(int(i) for i in in_range["infocode"].unique())
    return df


def _by_infocode(table: str, cols: str) -> Callable[[object, dict], pl.DataFrame]:
    def fetch(cur, ctx: dict) -> pl.DataFrame:
        df = query(
            cur,
            f"SELECT {cols} FROM {DS}.{table} WHERE infocode = ANY(%(ids)s)",
            {"ids": ctx["infocodes"]},
        )
        if table == "ds2cusipchg":
            ctx["cusips"] = sorted({c[:8] for c in df["cusip"].drop_nulls()})
        return df

    return fetch


def _crsp_security(cur, ctx: dict) -> pl.DataFrame:
    # Every security of each matched company (PERMCO), so a company's market cap can sum
    # share classes that are not on the list themselves.
    df = query(
        cur,
        "SELECT permno, permco, secinfostartdt, secinfoenddt, cusip, ticker, shareclass, "
        "sharetype, securitytype, securitysubtype FROM crsp.stksecurityinfohist "
        "WHERE permco IN (SELECT permco FROM crsp.stksecurityinfohist "
        "WHERE left(cusip, 8) = ANY(%(cusips)s))",
        {"cusips": ctx["cusips"]},
    )
    ctx["permnos"] = sorted(int(p) for p in df["permno"].unique())
    return df


def _by_permno(sql: str) -> Callable[[object, dict], pl.DataFrame]:
    def fetch(cur, ctx: dict) -> pl.DataFrame:
        return query(cur, sql, {"ids": ctx["permnos"], "start": ctx["start"]})

    return fetch


EXTRACTS: list[tuple[str, str, Callable[[object, dict], pl.DataFrame]]] = [
    ("ds_constituents", f"{DS}.ds2constmth", _constituents),
    (
        "ds_tickers",
        f"{DS}.ds2mnemchg",
        _by_infocode("ds2mnemchg", "infocode, startdate, enddate, ticker, dsmnem"),
    ),
    (
        "ds_cusips",
        f"{DS}.ds2cusipchg",
        _by_infocode("ds2cusipchg", "infocode, startdate, enddate, cusip"),
    ),
    (
        "ds_names",
        f"{DS}.wrds_ds_names",
        _by_infocode(
            "wrds_ds_names",
            "infocode, startdate, enddate, dscmpycode, isin, dsqtname, delistdate, "
            "ismajorsec, isprimqt",
        ),
    ),
    ("crsp_security", "crsp.stksecurityinfohist", _crsp_security),
    (
        "crsp_distributions",
        "crsp.stkdistributions",
        _by_permno(
            "SELECT permno, disexdt, distype, disdetailtype, disfacpr, disfacshr, disdivamt "
            "FROM crsp.stkdistributions WHERE permno = ANY(%(ids)s) AND disexdt >= %(start)s"
        ),
    ),
    (
        "crsp_delistings",
        "crsp.stkdelists",
        _by_permno(
            "SELECT permno, delistingdt, delactiontype, delstatustype, delreasontype, delret "
            "FROM crsp.stkdelists WHERE permno = ANY(%(ids)s)"
        ),
    ),
    (
        "crsp_daily",
        "crsp.dsf_v2",
        _by_permno(
            "SELECT permno, permco, dlycaldt, dlyret, dlyprc, dlyprevprc, dlycap, dlydelflg, "
            "securitytype, securitysubtype, sharetype FROM crsp.dsf_v2 "
            "WHERE permno = ANY(%(ids)s) AND dlycaldt >= %(start)s"
        ),
    ),
]


def fetch(cfg: dict, universe: str = "sp500") -> list[Path]:
    """Fetch every extract for membership universe `universe` and write each one.

    Rows start at the configured `data.start` (constituent spells that end before it are
    kept in `ds_constituents` but don't select securities).
    """
    spec = universes.spec(universe, CONFIG_DIR)
    if spec["kind"] != "membership" or spec.get("source") != "datastream":
        raise ValueError(f"universe {universe!r} is not a Datastream membership universe")
    start: date = cfg["data"]["start"]
    ctx = {"list_code": spec["list_code"], "start": start}
    fetched_at, sha = datetime.now(UTC).isoformat(), git_sha()
    frames: dict[str, tuple[str, pl.DataFrame]] = {}
    conn = connect()
    try:
        with conn.cursor() as cur:
            for name, table, fn in EXTRACTS:
                log.info("fetching %s from %s", name, table)
                frames[name] = (table, fn(cur, ctx))
                log.info("  %s rows", f"{frames[name][1].height:,}")
    finally:
        conn.close()
    dests = []
    for name, (table, df) in frames.items():
        dest = ROOT / cfg["data_root"] / "wrds" / name
        manifest = {
            "source": "wrds",
            "table": table,
            "universe": universe,
            "list_code": spec["list_code"],
            "start": str(start),
            "built_at": fetched_at,
            "git_sha": sha,
        }
        store.write(df, dest, manifest, file=f"{name}.parquet")
        dests.append(dest)
    return dests


def load(root: Path, name: str) -> tuple[pl.DataFrame, dict]:
    """One cached extract and its manifest; raises if `gnns fetch-wrds` hasn't run."""
    dest = root / "wrds" / name
    if not (dest / "manifest.json").exists():
        raise FileNotFoundError(f"{dest} missing; run gnns fetch-wrds first")
    lf, manifest = store.open(dest)
    return lf.collect(), manifest
