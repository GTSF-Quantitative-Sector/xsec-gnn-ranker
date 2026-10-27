"""Load a provenance-checked stream of known-at macro state observations."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

FACTORS = ("rates", "credit", "vix", "oil", "gold")
UNITS = {"rates": "percent", "credit": "percent", "vix": "index", "oil": "price", "gold": "price"}


def load_macro(root: Path, spec: dict) -> pl.DataFrame:
    """timestamp means availability, not observation date; no inferred publication lags.

    Each event updates the latest known state of one factor. Providers with revisions
    must first reconstruct that state; publishing an old observation as current is invalid.
    The sidecar records upstream verification, not a verification performed here.
    """
    path = root / spec["path"]
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("known_at_verified") is not True:
        raise ValueError(
            "macro source requires verified known-at timestamps; "
            "legacy date/value cache is exploratory"
        )
    if meta.get("units") != UNITS or not meta.get("availability_method") or not meta.get("source"):
        raise ValueError(
            "macro manifest needs source, availability_method and the documented factor units"
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta.get("sha256"):
        raise ValueError("macro source SHA-256 differs from its manifest")
    df = pl.read_csv(path)
    if set(df.columns) != {"timestamp", "factor", "value"}:
        raise ValueError("macro CSV must contain timestamp, factor, value")
    timestamps = []
    for value in df["timestamp"]:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            raise ValueError("macro timestamps must include an explicit UTC offset")
        timestamps.append(ts.astimezone(ZoneInfo("America/New_York")))
    df = df.with_columns(
        pl.Series("timestamp", timestamps, dtype=pl.Datetime("us", "America/New_York")),
        pl.col("value").cast(pl.Float64),
    )
    if df.is_empty() or df.select(pl.struct("timestamp", "factor").is_duplicated().any()).item():
        raise ValueError("empty macro source or duplicate timestamp/factor")
    if set(df["factor"]) != set(FACTORS) or not np.isfinite(df["value"].to_numpy()).all():
        raise ValueError("macro source must contain all five factors with finite values")
    return df.sort("timestamp", "factor")
