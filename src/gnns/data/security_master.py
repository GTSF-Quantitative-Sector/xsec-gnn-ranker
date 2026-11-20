"""Security master: one translation table from `security_id` (Datastream infocode) to every
other identifier, by date. Contract: docs/CONTRACTS.md, "Identifiers". Built in memory by
`gnns build-universe` (not stored); the universe table and raw bars carry its values.

Rows are `(security_id, id_type, value, valid_from, valid_to, source)`, dates inclusive,
`valid_to` null = still valid. Each `id_type` is derived by one rule (`derive`); reviewed
rows in `config/universes/overrides.csv` replace derived rows on the dates they
cover (`apply_overrides`); `violations` lists every broken invariant.
"""

from datetime import date, timedelta
from pathlib import Path

import polars as pl

COLUMNS = ["security_id", "id_type", "value", "valid_from", "valid_to", "source"]
SCHEMA = {
    "security_id": pl.String,
    "id_type": pl.String,
    "value": pl.String,
    "valid_from": pl.Date,
    "valid_to": pl.Date,
    "source": pl.String,
}
ID_TYPES = (
    "cusip",
    "isin",
    "company",
    "ticker_datastream",
    "permno",
    "permco",
    "ticker_crsp",
    "ticker_massive",
)
# A value of these points to at most one security on any date (company/permco may be shared
# by share classes; CRSP tickers are reference only).
UNIQUE = ("cusip", "isin", "permno", "ticker_datastream", "ticker_massive")
DS_OPEN_END = date(2079, 6, 5)  # Datastream's "still valid" end date
FAR = date(9999, 12, 31)
OVERRIDE_COLUMNS = ["security_id", "id_type", "value", "valid_from", "valid_to", "note"]


def empty() -> pl.DataFrame:
    return pl.DataFrame(schema=SCHEMA)


def security_ids(infocode: pl.Expr) -> pl.Expr:
    """Datastream infocode (read as a number) -> `security_id` string."""
    return infocode.cast(pl.Int64).cast(pl.String)


def _open_end(col: str) -> pl.Expr:
    return pl.when(pl.col(col) >= DS_OPEN_END).then(None).otherwise(pl.col(col))


def _datastream(df: pl.DataFrame, value: pl.Expr, id_type: str) -> pl.DataFrame:
    return df.select(
        security_id=security_ids(pl.col("infocode")),
        id_type=pl.lit(id_type),
        value=value.cast(pl.String),
        valid_from=pl.col("startdate"),
        valid_to=_open_end("enddate"),
        source=pl.lit("datastream"),
    ).drop_nulls(["security_id", "value", "valid_from"])


def derive(
    ds_names: pl.DataFrame,
    ds_tickers: pl.DataFrame,
    ds_cusips: pl.DataFrame,
    crsp_security: pl.DataFrame,
    crsp_last: date,
) -> pl.DataFrame:
    """Every derived link except `ticker_massive` (which needs Massive's bars).

    - `cusip`, `isin`, `company`: Datastream's dated rows.
    - `ticker_datastream`: Datastream's ticker history, cut at the security's Datastream
      delisting date (Datastream leaves a dead security's last ticker open).
    - `permno`, `permco`, `ticker_crsp`: the CRSP security whose 8-character CUSIP matches
      the security's CUSIP, on the dates both are valid, up to `crsp_last`.
    """
    delisted = (
        ds_names.select(security_id=security_ids(pl.col("infocode")), delist=pl.col("delistdate"))
        .filter(pl.col("delist").is_not_null() & (pl.col("delist") < DS_OPEN_END))
        .group_by("security_id")
        .agg(pl.col("delist").max())
    )
    tickers = (
        _datastream(ds_tickers, pl.col("ticker"), "ticker_datastream")
        .join(delisted, on="security_id", how="left")
        .filter(pl.col("delist").is_null() | (pl.col("valid_from") <= pl.col("delist")))
        .with_columns(
            valid_to=pl.when(
                pl.col("delist").is_not_null()
                & (pl.col("valid_to").is_null() | (pl.col("valid_to") > pl.col("delist")))
            )
            .then(pl.col("delist"))
            .otherwise(pl.col("valid_to"))
        )
        .drop("delist")
    )
    cusips = _datastream(ds_cusips, pl.col("cusip").str.slice(0, 8), "cusip")
    parts = [
        cusips,
        _datastream(ds_names, pl.col("isin"), "isin"),
        _datastream(ds_names, pl.col("dscmpycode").cast(pl.Int64), "company"),
        tickers,
    ]

    crsp = crsp_security.select(
        cusip8=pl.col("cusip").str.slice(0, 8),
        c_from=pl.col("secinfostartdt"),
        c_to=pl.min_horizontal(pl.col("secinfoenddt"), pl.lit(crsp_last)),
        permno=pl.col("permno").cast(pl.String),
        permco=pl.col("permco").cast(pl.String),
        ticker_crsp=pl.col("ticker"),
    )
    linked = (
        cusips.rename({"value": "cusip8"})
        .join(crsp, on="cusip8")
        .with_columns(
            valid_from=pl.max_horizontal("valid_from", "c_from"),
            valid_to=pl.min_horizontal(pl.col("valid_to").fill_null(FAR), "c_to"),
        )
        .filter(pl.col("valid_from") <= pl.col("valid_to"))
    )
    for id_type in ("permno", "permco", "ticker_crsp"):
        parts.append(
            linked.select(
                "security_id",
                id_type=pl.lit(id_type),
                value=pl.col(id_type),
                valid_from="valid_from",
                valid_to="valid_to",
                source=pl.lit("crsp"),
            ).drop_nulls("value")
        )
    return merge(pl.concat([p.select(COLUMNS) for p in parts]))


def merge(sm: pl.DataFrame) -> pl.DataFrame:
    """Join adjacent or overlapping rows that carry the same value, then sort.

    Rows with different values are left as they are, so overlaps between them stay
    visible to `violations`.
    """
    out: list[dict] = []
    rows = sm.sort("security_id", "id_type", "value", "valid_from").iter_rows(named=True)
    for r in rows:
        prev = out[-1] if out else None
        if (
            prev is not None
            and (prev["security_id"], prev["id_type"], prev["value"])
            == (r["security_id"], r["id_type"], r["value"])
            and (prev["valid_to"] is None or r["valid_from"] <= prev["valid_to"] + timedelta(1))
        ):
            if prev["valid_to"] is not None and (
                r["valid_to"] is None or r["valid_to"] > prev["valid_to"]
            ):
                prev["valid_to"] = r["valid_to"]
            continue
        out.append(dict(r))
    return pl.DataFrame(out, schema=SCHEMA).sort("security_id", "id_type", "valid_from")


def read_overrides(path: Path) -> pl.DataFrame:
    """The reviewed overrides file (`security_id, id_type, value, valid_from, valid_to,
    note`), as security-master rows with `source = override`; empty if the file is absent."""
    if not path.exists():
        return empty()
    df = pl.read_csv(path, schema_overrides={c: pl.String for c in OVERRIDE_COLUMNS})
    if missing := sorted(set(OVERRIDE_COLUMNS) - set(df.columns)):
        raise ValueError(f"{path}: missing columns {missing}")
    if bad := sorted(set(df["id_type"]) - set(ID_TYPES)):
        raise ValueError(f"{path}: unknown id_type {bad}; valid: {list(ID_TYPES)}")
    return df.select(
        "security_id",
        "id_type",
        "value",
        pl.col("valid_from").str.to_date(),
        pl.col("valid_to").str.to_date(strict=False),
        source=pl.lit("override"),
    )


def apply_overrides(sm: pl.DataFrame, overrides: pl.DataFrame) -> pl.DataFrame:
    """Override rows replace derived rows of the same security and id_type on the dates
    they cover (derived rows are cut around them). An override with an empty `value`
    only removes."""
    if overrides.is_empty():
        return sm
    rows = [dict(r) for r in sm.iter_rows(named=True)]
    for o in overrides.iter_rows(named=True):
        o_to = o["valid_to"] or FAR
        kept = []
        for r in rows:
            r_to = r["valid_to"] or FAR
            same = (r["security_id"], r["id_type"]) == (o["security_id"], o["id_type"])
            if not same or r_to < o["valid_from"] or r["valid_from"] > o_to:
                kept.append(r)
                continue
            if r["valid_from"] < o["valid_from"]:
                kept.append(r | {"valid_to": o["valid_from"] - timedelta(1)})
            if r_to > o_to:
                kept.append(r | {"valid_from": o_to + timedelta(1)})
        rows = kept
    added = overrides.filter(pl.col("value").is_not_null() & (pl.col("value") != ""))
    out = pl.concat([pl.DataFrame(rows, schema=SCHEMA), added.select(COLUMNS)])
    return out.sort("security_id", "id_type", "valid_from")


def violations(sm: pl.DataFrame) -> list[str]:
    """Every broken invariant: a security with two values of one id_type on a date, or a
    UNIQUE identifier pointing at two securities on a date. Messages name securities,
    id_types and dates, never values (some values are CRSP data)."""
    out = []
    s = sm.with_columns(end=pl.col("valid_to").fill_null(FAR))
    fwd = (
        s.sort("security_id", "id_type", "valid_from")
        .with_columns(prev_end=pl.col("end").shift(1).over("security_id", "id_type"))
        .filter(pl.col("valid_from") <= pl.col("prev_end"))
    )
    for r in fwd.iter_rows(named=True):
        out.append(f"security {r['security_id']}: two {r['id_type']} values on {r['valid_from']}")
    pairs = (
        s.filter(pl.col("id_type").is_in(UNIQUE))
        .join(s.filter(pl.col("id_type").is_in(UNIQUE)), on=["id_type", "value"], suffix="_b")
        .filter(
            (pl.col("security_id") < pl.col("security_id_b"))
            & (pl.col("valid_from") <= pl.col("end_b"))
            & (pl.col("valid_from_b") <= pl.col("end"))
        )
    )
    for r in pairs.iter_rows(named=True):
        on = max(r["valid_from"], r["valid_from_b"])
        out.append(
            f"securities {r['security_id']} and {r['security_id_b']} share a {r['id_type']} on {on}"
        )
    return out


def _valid_on(sm: pl.DataFrame, id_type: str, on: date) -> pl.DataFrame:
    return sm.filter(
        (pl.col("id_type") == id_type)
        & (pl.col("valid_from") <= on)
        & (pl.col("valid_to").is_null() | (pl.col("valid_to") >= on))
    )


def lookup(sm: pl.DataFrame, security_ids: list[str], id_type: str, on: date) -> pl.Series:
    """The `id_type` value of each security on date `on` (null where none is valid)."""
    valid = _valid_on(sm, id_type, on).select("security_id", "value")
    keys = pl.DataFrame({"security_id": security_ids}, schema={"security_id": pl.String})
    return keys.join(valid, on="security_id", how="left", maintain_order="left")["value"]


def resolve(sm: pl.DataFrame, id_type: str, values: list[str], on: date) -> pl.Series:
    """The security each `id_type` value points to on date `on` (null where none)."""
    valid = _valid_on(sm, id_type, on).select("value", "security_id")
    keys = pl.DataFrame({"value": values}, schema={"value": pl.String})
    return keys.join(valid, on="value", how="left", maintain_order="left")["security_id"]


def attach(
    df: pl.DataFrame, sm: pl.DataFrame, id_type: str, key: str = "ticker", date: str = "timestamp"
) -> pl.DataFrame:
    """`df` plus a column `id_type`: the value valid for each row's security on its date.

    `key` holds security ids; `date` may be a date or a datetime column.
    """
    links = sm.filter(pl.col("id_type") == id_type).select(
        pl.col("security_id").alias(key), "value", "valid_from", "valid_to"
    )
    rows = df.with_row_index("_row").with_columns(_day=pl.col(date).cast(pl.Date))
    hits = (
        rows.select("_row", key, "_day")
        .join(links, on=key)
        .filter(
            (pl.col("valid_from") <= pl.col("_day"))
            & (pl.col("valid_to").is_null() | (pl.col("valid_to") >= pl.col("_day")))
        )
        .select("_row", pl.col("value").alias(id_type))
        .unique("_row", keep="first")
    )
    return rows.join(hits, on="_row", how="left").sort("_row").drop("_row", "_day")
