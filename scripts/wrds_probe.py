"""Find out what our WRDS subscription can read (issue #1, step 1). Run it yourself.

Prints metadata only: which libraries are visible, which candidate tables exist and are
readable, each table's date range, table names that look like index membership or fund
holdings, column names of the key CRSP/Datastream tables, and aggregate checks of whether
Datastream's monthly S&P 500 constituent list (ds2constmth) can serve as our point-in-time
universe. It never prints or saves a data row or identifier, so the output can be shared
with a coding assistant, unlike WRDS data itself.

Setup: README, "WRDS access". Results go to <data_root>/wrds_probe/ (delete it when done).

    python scripts/wrds_probe.py                       # all sections
    python scripts/wrds_probe.py --all-tables          # also list every table name per library
    python scripts/wrds_probe.py --sections datastream indexes   # skip the general section

Sections: `general` (libraries, candidate tables, table-name search), `datastream`
(columns + Datastream S&P 500 list checks), `indexes` (other S&P 500 constituent sources:
Compustat, CRSP index tables, Capital IQ).
"""

import argparse
import calendar
import json
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql

from gnns.config import load_config

ROOT = Path(__file__).resolve().parents[1]
HOST, PORT, DBNAME = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"
TIMEOUT_MS = 180_000  # per query; a slow table reports a timeout instead of hanging

# (library, table, date columns to range over, optional filter that keeps the query on an
# index for huge daily tables). Names that don't exist under our subscription are
# reported as missing, so guesses here are cheap. AAPL is PERMNO 14593, GVKEY 001690.
CANDIDATES: list[tuple[str, str, list[str], str | None]] = [
    # CRSP, current CIZ format
    ("crsp", "dsf_v2", ["dlycaldt"], "permno = 14593"),
    ("crsp", "stksecurityinfohist", ["secinfostartdt", "secinfoenddt"], None),
    ("crsp", "stkdelists", ["delistingdt"], None),
    ("crsp", "stkdistributions", ["disexdt"], None),
    ("crsp", "dsp500list_v2", ["mbrstartdt", "mbrenddt"], None),
    # CRSP S&P 500 membership lives in the CRSP Indexes product, which is not on our
    # product list; probe it anyway in case it is visible.
    ("crsp_a_indexes", "dsp500list_v2", ["mbrstartdt", "mbrenddt"], None),
    ("crsp_a_indexes", "dsp500list", ["start", "ending"], None),
    ("crsp_q_indexes", "dsp500list_v2", ["mbrstartdt", "mbrenddt"], None),
    # CRSP, legacy format
    ("crsp", "dsf", ["date"], "permno = 14593"),
    ("crsp", "dsi", ["date"], None),
    ("crsp", "dsp500list", ["start", "ending"], None),
    ("crsp", "dsedelist", ["dlstdt"], None),
    ("crsp", "dsedist", ["exdt"], None),
    ("crsp", "stocknames", ["namedt", "nameenddt"], None),
    # CRSP/Compustat link
    ("crsp", "ccmxpf_lnkhist", ["linkdt", "linkenddt"], None),
    ("crsp_a_ccm", "ccmxpf_lnkhist", ["linkdt", "linkenddt"], None),
    # Compustat
    ("comp", "idxcst_his", ["from", "thru"], None),
    ("comp", "funda", ["datadate"], None),
    ("comp", "company", [], None),
    ("comp", "secd", ["datadate"], "gvkey = '001690'"),
    ("comp", "co_hgic", ["indfrom", "indthru"], None),
    ("comp_segments_hist_daily", "wrds_segmerged", ["datadate"], None),
    ("comp_segments_hist_daily", "seg_customer", ["srcdate"], None),
    # Fund holdings: S&P 500 index funds/ETFs as a membership proxy. The holdings tables
    # have hundreds of millions of rows, so only check that they are readable.
    ("crsp_q_mutualfunds", "holdings", [], None),
    ("crsp_q_mutualfunds", "holdings_co_info", [], None),
    ("crsp_q_mutualfunds", "fund_hdr", [], None),
    ("crsp_q_mutualfunds", "portnomap", ["begdt", "enddt"], None),
    ("tr_mutualfunds", "s12", [], None),
]

# Table names worth a look in any visible library (case-insensitive substrings).
NAME_PATTERNS = ["500", "const", "idx", "index", "member", "holding"]


def connect() -> "psycopg2.extensions.connection":
    load_dotenv(ROOT / ".env")
    user = os.environ.get("WRDS_USERNAME")
    if not user:
        sys.exit("Set WRDS_USERNAME in .env (see README, 'WRDS access').")
    # None is dropped by psycopg2, and libpq then falls back to the pgpass file.
    password = os.environ.get("WRDS_PASSWORD") or None
    print(f"Connecting to WRDS as {user} (approve the Duo push if your phone asks)...")
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
        sys.exit(
            f"Could not connect: {e}\nCheck WRDS_USERNAME and WRDS_PASSWORD in .env "
            "(see README, 'WRDS access')."
        )
    conn.autocommit = True  # a failed probe must not abort the ones after it
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {TIMEOUT_MS}")
    return conn


def libraries(cur) -> list[str]:
    """Schemas that have tables and that we have USAGE on."""
    cur.execute(
        "SELECT DISTINCT table_schema FROM information_schema.tables "
        "WHERE has_schema_privilege(table_schema, 'USAGE') "
        "AND table_schema NOT IN ('information_schema', 'pg_catalog') ORDER BY 1"
    )
    return [r[0] for r in cur.fetchall()]


def tables(cur, library: str) -> list[str]:
    cur.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = %s ORDER BY 1",
        (library,),
    )
    return [r[0] for r in cur.fetchall()]


def name_matches(cur, libs: list[str]) -> dict[str, list[str]]:
    """Table names in the visible libraries that contain any of NAME_PATTERNS."""
    cur.execute(
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_schema = ANY(%s) AND lower(table_name) LIKE ANY(%s) ORDER BY 1, 2",
        (libs, [f"%{p}%" for p in NAME_PATTERNS]),
    )
    found: dict[str, list[str]] = {}
    for lib, name in cur.fetchall():
        found.setdefault(lib, []).append(name)
    return found


def columns(cur, library: str, table: str) -> list[str]:
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
        (library, table),
    )
    return [r[0] for r in cur.fetchall()]


def probe(cur, library: str, table: str, date_cols: list[str], where: str | None) -> dict:
    cols = columns(cur, library, table)
    if not cols:
        return {"status": "missing"}
    ident = sql.Identifier(library, table)
    try:
        cur.execute(sql.SQL("SELECT 1 FROM {} LIMIT 1").format(ident))
        cur.fetchall()  # discarded: only whether the read succeeds matters
    except psycopg2.Error as e:
        return {"status": "no access", "error": e.pgcode or type(e).__name__}
    result: dict = {"status": "readable", "n_columns": len(cols)}
    for col in [c for c in date_cols if c in cols]:
        query = sql.SQL("SELECT min({c}), max({c}) FROM {t}").format(c=sql.Identifier(col), t=ident)
        if where:
            query += sql.SQL(" WHERE " + where)  # constant from CANDIDATES, never user input
        try:
            cur.execute(query)
            lo, hi = cur.fetchone()
            result[col] = [str(lo), str(hi)]
        except psycopg2.Error as e:
            result[col] = f"failed ({e.pgcode or type(e).__name__})"
    if where:
        result["date_range_filter"] = where
    return result


# --- Datastream S&P 500 constituents -------------------------------------------------------
#
# Can tr_ds_equities.ds2constmth serve as our point-in-time S&P 500 list? Everything below
# is an aggregate (counts, min/max dates, index names); no membership row or identifier is
# printed.

DS = "tr_ds_equities"
# S&P 500 list code used by github.com/ariusmak/krauss-ml-statarb; checked against
# ds2indexlist below.
SP500_DS_CODE = 4408
CRSP_LAST = "2025-12-31"  # last date in crsp.dsf_v2 (annual update)
WINDOW_START = "2009-01-30"  # first Datastream list date of our window
# Tables whose column names we want to see (to check what CRSP and Datastream offer).
COLUMN_TABLES = [
    ("crsp", "dsf_v2"),
    ("crsp", "stksecurityinfohist"),
    ("crsp", "stkdelists"),
    ("crsp", "stkdistributions"),
    (DS, "ds2indexlist"),
    (DS, "ds2constmth"),
    (DS, "ds2constdly"),
    (DS, "ds2cusipchg"),
    (DS, "wrds_ds_names"),
]


def run(cur, query: sql.Composable | str, params: tuple | None = None) -> list[tuple] | str:
    """Rows of an aggregate query, or 'failed (<code>)'."""
    try:
        cur.execute(query, params)
        return cur.fetchall()
    except psycopg2.Error as e:
        return f"failed ({e.pgcode or type(e).__name__})"


def show(title: str, rows: list[tuple] | str, header: tuple[str, ...] = ()) -> None:
    print(f"\n  {title}")
    if isinstance(rows, str):
        print(f"    {rows}")
        return
    if header:
        print("    " + " | ".join(header))
    for r in rows:
        print("    " + " | ".join("" if v is None else str(v) for v in r))


def check_dates() -> list[date]:
    """Year-ends 2009-2025, then every month-end of the current year that has passed."""
    today = date.today()
    out = [date(y, 12, 31) for y in range(2009, today.year)]
    for m in range(1, today.month):
        out.append(date(today.year, m, calendar.monthrange(today.year, m)[1]))
    return out


def datastream(cur) -> dict:
    report: dict = {}
    t = {name: sql.Identifier(DS, name) for name in ("ds2indexlist", "ds2constmth", "ds2constdly")}
    code = SP500_DS_CODE

    print("\nColumns (for checking what each table offers):")
    cols = {f"{lib}.{tab}": columns(cur, lib, tab) for lib, tab in COLUMN_TABLES}
    report["columns"] = cols
    for name, names in cols.items():
        print(f"  {name}: {', '.join(names) if names else '(missing)'}")

    print(f"\nDatastream S&P 500 constituents ({DS}):")

    # 1. Which index lists are S&P 500 lists, and at what license level?
    il_cols = [
        c
        for c in ("indexlistmnem", "indexlistdesc", "permitlevelcode")
        if c in cols[f"{DS}.ds2indexlist"]
    ]
    text_cols = [c for c in ("indexlistmnem", "indexlistdesc") if c in il_cols]
    sp_codes: list[int] = [code]
    if text_cols:
        where = sql.SQL(" OR ").join(
            sql.SQL("{} ILIKE %s").format(sql.Identifier(c)) for c in text_cols
        )
        rows = run(
            cur,
            sql.SQL(
                "SELECT indexlistintcode, {cols} FROM {t} WHERE indexlistintcode = %s OR "
            ).format(cols=sql.SQL(", ").join(map(sql.Identifier, il_cols)), t=t["ds2indexlist"])
            + where
            + sql.SQL(" ORDER BY 1 LIMIT 60"),
            (code, *["%S&P 500%" if c == "indexlistdesc" else "%S&PCOMP%" for c in text_cols]),
        )
        show("Index lists named like the S&P 500", rows, ("code", *il_cols))
        report["sp500_lists"] = rows
        if isinstance(rows, list):
            sp_codes = sorted({code, *(r[0] for r in rows)})

    # 2. What the daily table (ds2constdly) holds.
    names = sql.SQL(", ").join(sql.SQL("l.") + sql.Identifier(c) for c in il_cols)
    rows = run(
        cur,
        sql.SQL(
            "SELECT c.indexlistintcode, {names}, count(*), count(DISTINCT c.infocode), "
            "min(c.startdate), max(c.enddate) FROM {d} c LEFT JOIN {l} l USING (indexlistintcode) "
            "GROUP BY c.indexlistintcode, {names} ORDER BY count(*) DESC"
        ).format(names=names, d=t["ds2constdly"], l=t["ds2indexlist"])
        if il_cols
        else sql.SQL(
            "SELECT indexlistintcode, count(*), count(DISTINCT infocode), min(startdate), "
            "max(enddate) FROM {d} GROUP BY 1 ORDER BY 2 DESC"
        ).format(d=t["ds2constdly"]),
    )
    show(
        "Lists in ds2constdly (the 790-row daily table)",
        rows,
        ("code", *il_cols, "rows", "stocks", "first start", "last end"),
    )
    report["constdly_lists"] = rows

    # 3. Monthly-table summary for every S&P 500-like list found above.
    rows = run(
        cur,
        sql.SQL(
            "SELECT indexlistintcode, count(*), count(DISTINCT infocode), min(startdate), "
            "max(startdate), max(enddate) FROM {m} WHERE indexlistintcode = ANY(%s) "
            "GROUP BY 1 ORDER BY 1"
        ).format(m=t["ds2constmth"]),
        (sp_codes,),
    )
    show(
        "ds2constmth summary for those lists",
        rows,
        ("code", "spells", "stocks", "first start", "last start", "last end"),
    )
    report["constmth_lists"] = rows

    # 4. Detail for the S&P 500 list itself.
    m = t["ds2constmth"]
    rows = run(
        cur,
        sql.SQL(
            "SELECT count(*), count(DISTINCT infocode), count(*) FILTER (WHERE infocode IS NULL), "
            "count(*) FILTER (WHERE enddate IS NULL), "
            "(SELECT count(*) FROM (SELECT infocode FROM {m} WHERE indexlistintcode = %s "
            "  GROUP BY infocode HAVING count(*) > 1) x) "
            "FROM {m} WHERE indexlistintcode = %s"
        ).format(m=m),
        (code, code),
    )
    show(
        f"List {code}: spell counts",
        rows,
        ("spells", "stocks", "null infocode", "open (no end)", "stocks with re-entries"),
    )
    report["sp500_counts"] = rows

    for col in ("startdate", "enddate"):
        c = sql.Identifier(col)
        rows = run(
            cur,
            sql.SQL(
                "SELECT count(*) FILTER (WHERE {c} = (date_trunc('month', {c}) "
                "  + interval '1 month - 1 day')::date), "
                "count(*) FILTER (WHERE extract(isodow FROM {c}) < 6 AND {c} + 7 > "
                "  (date_trunc('month', {c}) + interval '1 month - 1 day')::date), "
                "count(*) FROM {m} WHERE indexlistintcode = %s AND {c} IS NOT NULL"
            ).format(c=c, m=m),
            (code,),
        )
        show(
            f"List {code}: is {col} a month-end? (exact dates would spread over the month)",
            rows,
            ("last calendar day", "weekday in last 7 days", "all"),
        )
        report[f"sp500_{col}_month_end"] = rows
        rows = run(
            cur,
            sql.SQL(
                "SELECT extract(day FROM {c})::int, count(*) FROM {m} "
                "WHERE indexlistintcode = %s AND {c} IS NOT NULL GROUP BY 1 ORDER BY 1"
            ).format(c=c, m=m),
            (code,),
        )
        report[f"sp500_{col}_day_of_month"] = rows
        if isinstance(rows, list):
            print(f"    day of month -> count: {', '.join(f'{d}:{n}' for d, n in rows)}")

    # Index changes per year: S&P makes roughly 20-30 a year, so starts should look similar.
    rows = run(
        cur,
        sql.SQL(
            "SELECT y, sum(s), sum(e) FROM ("
            "  SELECT extract(year FROM startdate)::int y, 1 s, 0 e FROM {m} "
            "  WHERE indexlistintcode = %s"
            "  UNION ALL SELECT extract(year FROM enddate)::int, 0, 1 FROM {m} "
            "  WHERE indexlistintcode = %s AND enddate IS NOT NULL"
            ") x WHERE y >= 2000 GROUP BY y ORDER BY y"
        ).format(m=m),
        (code, code),
    )
    show(f"List {code}: spell starts and ends per year (2000+)", rows, ("year", "starts", "ends"))
    report["sp500_changes_per_year"] = rows

    # Members on each check date: should be about 500-505 (share classes count twice).
    rows = run(
        cur,
        sql.SQL(
            "SELECT d, count(c.startdate) FROM unnest(%s::date[]) AS d LEFT JOIN {m} c "
            "ON c.indexlistintcode = %s AND c.startdate <= d "
            "AND (c.enddate >= d OR c.enddate IS NULL) "
            "GROUP BY d ORDER BY d"
        ).format(m=m),
        (check_dates(), code),
    )
    show(f"List {code}: members on each date", rows, ("date", "members"))
    report["sp500_members_by_date"] = rows

    # 5. Can the list's infocodes be linked to CRSP PERMNOs by CUSIP?
    cusip_cols = cols[f"{DS}.ds2cusipchg"]
    crsp_cols = cols["crsp.stksecurityinfohist"]
    crsp_cusip = next((c for c in ("cusip", "hdrcusip", "cusip9") if c in crsp_cols), None)
    if {"infocode", "cusip"} <= set(cusip_cols) and crsp_cusip and "permno" in crsp_cols:
        rows = run(
            cur,
            sql.SQL(
                "WITH sp AS (SELECT DISTINCT infocode FROM {m} "
                "  WHERE indexlistintcode = %s AND infocode IS NOT NULL), "
                "dc AS (SELECT DISTINCT c.infocode, left(c.cusip, 8) AS cusip8 "
                "  FROM {cc} c JOIN sp USING (infocode)), "
                "cr AS (SELECT DISTINCT left({col}, 8) AS cusip8, permno FROM {ci} "
                "  WHERE {col} IS NOT NULL), "
                "mt AS (SELECT dc.infocode, cr.permno FROM dc JOIN cr USING (cusip8)) "
                "SELECT (SELECT count(*) FROM sp), (SELECT count(DISTINCT infocode) FROM dc), "
                "(SELECT count(DISTINCT infocode) FROM mt), "
                "(SELECT count(*) FROM (SELECT infocode FROM mt GROUP BY infocode "
                "  HAVING count(DISTINCT permno) > 1) x)"
            ).format(
                m=m,
                cc=sql.Identifier(DS, "ds2cusipchg"),
                col=sql.Identifier(crsp_cusip),
                ci=sql.Identifier("crsp", "stksecurityinfohist"),
            ),
            (code,),
        )
        show(
            f"List {code}: link to CRSP by 8-character CUSIP (any date)",
            rows,
            ("stocks", "with a CUSIP", "matched to a PERMNO", "matched to 2+ PERMNOs"),
        )
        report["sp500_crsp_link"] = rows
    else:
        print("\n  CUSIP link skipped: needed columns not found (see the column lists above).")

    # 6. Members on the list dates themselves. Spells run from the first to the last
    # month-end list a stock is on, so calendar month-ends that fall on a weekend undercount.
    rows = run(
        cur,
        sql.SQL(
            "WITH sp AS (SELECT startdate, enddate FROM {m} WHERE indexlistintcode = %s), "
            "d AS (SELECT startdate AS d FROM sp UNION SELECT enddate FROM sp), "
            "c AS (SELECT d.d, count(*) AS n FROM d JOIN sp "
            "  ON sp.startdate <= d.d AND sp.enddate >= d.d WHERE d.d >= '2009-01-01' "
            "  GROUP BY d.d) "
            "SELECT count(*), min(d), max(d), min(n), max(n), round(avg(n), 1), "
            "count(*) FILTER (WHERE n < 495 OR n > 510), "
            "(SELECT count(*) FROM (SELECT 1 FROM c GROUP BY date_trunc('month', d) "
            "  HAVING count(*) > 1) x) FROM c"
        ).format(m=m),
        (code,),
    )
    show(
        f"List {code}: members on each list date since 2009",
        rows,
        (
            "list dates",
            "first",
            "last",
            "min members",
            "max members",
            "mean",
            "dates outside 495-510",
            "months with 2+ list dates",
        ),
    )
    report["sp500_members_on_list_dates"] = rows

    # 7. Date-aware link: CUSIP valid on a date in Datastream, matched to the CRSP security
    # whose CUSIP is valid on that date. CRSP ends at CRSP_LAST, so later spells are counted
    # separately (they need another route to a ticker). Run twice: at every spell's start,
    # and for the spells our 2009+ window uses, at max(spell start, first 2009 list date).
    for label, floor in (("all spells, at spell start", "1900-01-01"), ("2009+", WINDOW_START)):
        rows = run(
            cur,
            sql.SQL(
                "WITH sp AS (SELECT infocode, greatest(startdate, %s::date) AS d FROM {m} "
                "  WHERE indexlistintcode = %s AND infocode IS NOT NULL AND enddate >= %s::date "
                "  AND greatest(startdate, %s::date) <= %s), "
                "dc AS (SELECT sp.infocode, sp.d, left(c.cusip, 8) AS cusip8 FROM sp JOIN {cc} c "
                "  ON c.infocode = sp.infocode AND (c.startdate <= sp.d OR c.startdate IS NULL) "
                "  AND (c.enddate >= sp.d OR c.enddate IS NULL)), "
                "mt AS (SELECT dc.infocode, dc.d, count(DISTINCT s.permno) AS np FROM dc "
                "  LEFT JOIN {ci} s ON left(s.cusip, 8) = dc.cusip8 "
                "  AND s.secinfostartdt <= dc.d AND s.secinfoenddt >= dc.d "
                "  GROUP BY dc.infocode, dc.d) "
                "SELECT (SELECT count(*) FROM sp), count(*), count(*) FILTER (WHERE np = 1), "
                "count(*) FILTER (WHERE np = 0), count(*) FILTER (WHERE np > 1), "
                "(SELECT count(*) FROM {m} WHERE indexlistintcode = %s AND startdate > %s) "
                "FROM mt"
            ).format(
                m=m,
                cc=sql.Identifier(DS, "ds2cusipchg"),
                ci=sql.Identifier("crsp", "stksecurityinfohist"),
            ),
            (floor, code, floor, floor, CRSP_LAST, code, CRSP_LAST),
        )
        show(
            f"List {code}: date-aware CUSIP link to CRSP ({label})",
            rows,
            (
                f"spells to {CRSP_LAST}",
                "with a CUSIP then",
                "1 PERMNO",
                "no PERMNO",
                "2+ PERMNOs",
                f"spells after {CRSP_LAST}",
            ),
        )
        report[f"sp500_crsp_link_dated {label}"] = rows

    # Spells after CRSP's last date: does Datastream give them a ticker (route to Massive)?
    rows = run(
        cur,
        sql.SQL(
            "SELECT count(DISTINCT sp.infocode), count(DISTINCT n.infocode) FROM {m} sp "
            "LEFT JOIN {n} n ON n.infocode = sp.infocode AND n.ticker IS NOT NULL "
            "WHERE sp.indexlistintcode = %s AND sp.startdate > %s"
        ).format(m=m, n=sql.Identifier(DS, "wrds_ds_names")),
        (code, CRSP_LAST),
    )
    show(
        f"List {code}: stocks first listed after {CRSP_LAST} that have a Datastream ticker",
        rows,
        ("stocks", "with a ticker"),
    )
    report["sp500_after_crsp_tickers"] = rows
    return report


# --- Other S&P 500 constituent sources -----------------------------------------------------
#
# Settled by the 2026-11-17 run: every CRSP index table is no access, and Compustat's
# idxcst_his/indexcst_his hold only today's 503 S&P 500 members (no removed stocks), so they
# are survivorship-biased. Still open: Capital IQ's S&P 500 constituents (exact from/to
# dates, if they keep removed stocks) and comp.spidx_cst (dated snapshots?).

CIQ_SP500 = 2668699  # ciq.ciqindex "S&P 500"
SPIDX_SP500 = "500"  # comp.idx_index.indexid of the S&P 500
CIQ_LIBS = ["ciq", "ciq_common"]


def first(cols: list[str], *names: str) -> str | None:
    return next((n for n in names if n in cols), None)


def constituent_stats(cur, lib: str, tab: str, cols: list[str], key: str, value: str | int) -> dict:
    """The Datastream checks, for a constituent history table filtered to one index."""
    out: dict = {}
    frm = first(cols, "from", "fromdate", "startdate", "mbrstartdt")
    thru = first(cols, "thru", "thrudate", "todate", "enddate", "mbrenddt")
    alt = first(cols, "permno", "infocode", "tradingitemid", "constituentid")
    sec = [c for c in ("gvkey", "iid") if c in cols] or ([alt] if alt else [])
    if not frm or not thru or not sec:
        print(f"    skipped: no start/end/security columns in {lib}.{tab}")
        return out
    t, f, e, k = (
        sql.Identifier(lib, tab),
        sql.Identifier(frm),
        sql.Identifier(thru),
        sql.Identifier(key),
    )
    secs = sql.SQL(", ").join(map(sql.Identifier, sec))
    name = f"{lib}.{tab} [{key} = {value}]"

    rows = run(
        cur,
        sql.SQL(
            "SELECT count(*), (SELECT count(*) FROM (SELECT DISTINCT {secs} FROM {t} "
            "  WHERE {k} = %s) x), min({f}), max({f}), max({e}), "
            "count(*) FILTER (WHERE {e} IS NULL), "
            "(SELECT count(*) FROM (SELECT 1 FROM {t} WHERE {k} = %s GROUP BY {secs} "
            "  HAVING count(*) > 1) y) FROM {t} WHERE {k} = %s"
        ).format(secs=secs, t=t, k=k, f=f, e=e),
        (value, value, value),
    )
    show(
        f"{name}: spell counts",
        rows,
        ("spells", "securities", "first start", "last start", "last end", "open", "re-entries"),
    )
    out["counts"] = rows

    for col, ident in ((frm, f), (thru, e)):
        rows = run(
            cur,
            sql.SQL(
                "SELECT count(*) FILTER (WHERE extract(isodow FROM {c}) < 6 AND {c} + 7 > "
                "  (date_trunc('month', {c}) + interval '1 month - 1 day')::date), "
                "count(DISTINCT extract(day FROM {c})), count(*) "
                "FROM {t} WHERE {k} = %s AND {c} IS NOT NULL AND {c} >= '2000-01-01'"
            ).format(c=ident, t=t, k=k),
            (value,),
        )
        show(
            f"{name}: {col} since 2000 (exact dates spread over the month)",
            rows,
            ("weekday in month's last 7 days", "distinct days of month", "all"),
        )
        out[f"{col}_precision"] = rows

    rows = run(
        cur,
        sql.SQL(
            "SELECT y, sum(s), sum(n) FROM ("
            "  SELECT extract(year FROM {f})::int y, 1 s, 0 n FROM {t} WHERE {k} = %s"
            "  UNION ALL SELECT extract(year FROM {e})::int, 0, 1 FROM {t} "
            "  WHERE {k} = %s AND {e} IS NOT NULL"
            ") x WHERE y >= 2000 GROUP BY y ORDER BY y"
        ).format(f=f, e=e, t=t, k=k),
        (value, value),
    )
    show(f"{name}: spell starts and ends per year (2000+)", rows, ("year", "starts", "ends"))
    out["changes_per_year"] = rows

    dates = [*check_dates(), date.today()]
    rows = run(
        cur,
        sql.SQL(
            "SELECT d, count(c.{f}) FROM unnest(%s::date[]) AS d LEFT JOIN {t} c "
            "ON c.{k} = %s AND c.{f} <= d AND (c.{e} >= d OR c.{e} IS NULL) GROUP BY d ORDER BY d"
        ).format(f=f, e=e, t=t, k=k),
        (dates, value),
    )
    show(f"{name}: members on each date", rows, ("date", "members"))
    out["members_by_date"] = rows
    return out


def index_sources(cur) -> dict:
    report: dict = {}
    print("\nOther S&P 500 constituent sources:")

    # 1. Capital IQ's S&P 500 list: full history (closed spells, re-entries) and exact dates?
    cols = columns(cur, "ciq", "ciqindexconstituent")
    if cols:
        report["ciq stats"] = constituent_stats(
            cur, "ciq", "ciqindexconstituent", cols, "indexid", CIQ_SP500
        )
    else:
        print("  ciq.ciqindexconstituent: missing")

    # 2. comp.spidx_cst: which indexes it covers, and whether the S&P 500 has dated snapshots.
    rows = run(
        cur,
        "SELECT indexid, count(*), count(DISTINCT datadate), min(datadate), max(datadate) "
        "FROM comp.spidx_cst GROUP BY indexid ORDER BY 2 DESC LIMIT 15",
    )
    show(
        "comp.spidx_cst: largest indexes",
        rows,
        ("indexid", "rows", "dates", "first date", "last date"),
    )
    report["spidx indexes"] = rows
    rows = run(
        cur,
        "WITH c AS (SELECT datadate, count(*) AS n FROM comp.spidx_cst WHERE indexid = %s "
        "  GROUP BY datadate) "
        "SELECT count(*), min(datadate), max(datadate), min(n), max(n), round(avg(n), 1), "
        "count(*) FILTER (WHERE datadate >= '2009-01-01') FROM c",
        (SPIDX_SP500,),
    )
    show(
        f"comp.spidx_cst [indexid = {SPIDX_SP500}]: snapshots",
        rows,
        ("dates", "first", "last", "min rows", "max rows", "mean rows", "dates since 2009"),
    )
    report["spidx sp500"] = rows

    # 3. Capital IQ identifier tables: is there a bridge from tradingitemid/securityid to
    # CUSIP, GVKEY or ticker (and so to CRSP PERMNO or Massive)?
    rows = run(
        cur,
        "SELECT table_schema || '.' || table_name, string_agg(column_name, ', ' "
        "  ORDER BY ordinal_position) FROM information_schema.columns "
        "WHERE table_schema = ANY(%s) GROUP BY table_schema, table_name "
        "HAVING bool_or(column_name IN ('tradingitemid', 'securityid', 'companyid')) "
        "AND bool_or(column_name ILIKE '%%cusip%%' OR column_name ILIKE '%%gvkey%%' "
        "  OR column_name ILIKE '%%ticker%%' OR column_name ILIKE '%%isin%%') "
        "ORDER BY 1 LIMIT 40",
        (CIQ_LIBS,),
    )
    show(
        "Capital IQ tables that map its IDs to CUSIP/GVKEY/ticker/ISIN", rows, ("table", "columns")
    )
    report["ciq id tables"] = rows

    # 4. If a CIQ table maps securityid -> cusip, measure the S&P 500 list's match to CRSP.
    bridge = None
    if isinstance(rows, list):
        for name, colstr in rows:
            names = colstr.split(", ")
            if "securityid" in names and "cusip" in names:
                bridge = name
                break
    ti = columns(cur, "ciq", "ciqtradingitem")
    if bridge and {"tradingitemid", "securityid"} <= set(ti):
        lib, tab = bridge.split(".")
        rows = run(
            cur,
            sql.SQL(
                "WITH sp AS (SELECT DISTINCT tradingitemid FROM ciq.ciqindexconstituent "
                "  WHERE indexid = %s AND tradingitemid IS NOT NULL), "
                "cu AS (SELECT DISTINCT sp.tradingitemid, left(b.cusip, 8) AS cusip8 FROM sp "
                "  JOIN ciq.ciqtradingitem t USING (tradingitemid) "
                "  JOIN {b} b ON b.securityid = t.securityid WHERE b.cusip IS NOT NULL), "
                "mt AS (SELECT cu.tradingitemid, count(DISTINCT c.permno) AS np FROM cu "
                "  LEFT JOIN crsp.stksecurityinfohist c ON left(c.cusip, 8) = cu.cusip8 "
                "  GROUP BY cu.tradingitemid) "
                "SELECT (SELECT count(*) FROM sp), count(*), count(*) FILTER (WHERE np = 1), "
                "count(*) FILTER (WHERE np = 0), count(*) FILTER (WHERE np > 1) FROM mt"
            ).format(b=sql.Identifier(lib, tab)),
            (CIQ_SP500,),
        )
        show(
            f"CIQ S&P 500: link to CRSP by 8-character CUSIP via {bridge} (any date)",
            rows,
            ("trading items", "with a CUSIP", "1 PERMNO", "no PERMNO", "2+ PERMNOs"),
        )
        report["ciq sp500 crsp link"] = rows
    else:
        print(
            "\n  CIQ -> CRSP link skipped: no securityid -> cusip table found, or "
            "ciq.ciqtradingitem lacks tradingitemid/securityid "
            f"(ciqtradingitem columns: {', '.join(ti) or 'missing'})."
        )
    return report


def general(cur, report: dict, all_tables: bool) -> None:
    """Visible libraries, candidate tables and table-name matches."""
    libs = libraries(cur)
    report["libraries"] = libs
    print(f"\n{len(libs)} libraries visible.")
    if all_tables:
        report["tables"] = {lib: tables(cur, lib) for lib in libs}
    report["candidates"] = {}
    for library, table, date_cols, where in CANDIDATES:
        name = f"{library}.{table}"
        print(f"  {name:<45}", end="", flush=True)
        res = probe(cur, library, table, date_cols, where)
        report["candidates"][name] = res
        ranges = "  ".join(
            f"{c}: {res[c][0]}..{res[c][1]}" if isinstance(res[c], list) else f"{c}: {res[c]}"
            for c in date_cols
            if c in res
        )
        print(f"{res['status']:<11}{ranges}")

    matches = name_matches(cur, libs)
    report["name_matches"] = matches
    print(f"\nTables whose names contain any of {NAME_PATTERNS}:")
    for lib, names in matches.items():
        print(f"  {lib}: {', '.join(names)}")


SECTIONS = ["general", "datastream", "indexes"]


def main() -> None:
    parser = argparse.ArgumentParser(description="List what our WRDS subscription can read.")
    parser.add_argument("--all-tables", action="store_true", help="list every table name")
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=SECTIONS,
        default=SECTIONS,
        help="which sections to run (default: all)",
    )
    args = parser.parse_args()

    cfg = load_config()  # only for data_root
    report: dict = {"probed_at": datetime.now(UTC).isoformat(timespec="seconds")}
    # Not `with conn:` — since psycopg2 2.9 that opens a transaction even in autocommit
    # mode, so one refused query would abort every probe after it.
    conn = connect()
    try:
        with conn.cursor() as cur:
            if "general" in args.sections:
                general(cur, report, all_tables=args.all_tables)
            if "datastream" in args.sections:
                report["datastream"] = datastream(cur)
            if "indexes" in args.sections:
                report["indexes"] = index_sources(cur)
    finally:
        conn.close()

    suffix = "" if args.sections == SECTIONS else "_" + "_".join(args.sections)
    out = Path(cfg["data_root"]) / "wrds_probe" / f"probe{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved to {out}. It holds metadata only; read it before sharing.")


if __name__ == "__main__":
    main()
