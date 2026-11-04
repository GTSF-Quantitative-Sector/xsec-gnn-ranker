"""Load a provenance-checked stream of known-at macro state observations."""

import hashlib
from pathlib import Path

import numpy as np
import polars as pl

from gnns.data import store

FACTORS = ("rates", "credit", "vix", "oil", "gold")
UNITS = {"rates": "percent", "credit": "percent", "vix": "index", "oil": "price", "gold": "price"}


def load_macro(root: Path, spec: dict) -> tuple[pl.DataFrame, dict]:
    """timestamp means availability, not observation date; no inferred publication lags.

    `spec["path"]` is a dataset folder under `root` (docs/CONTRACTS.md, "Storage layout").
    Each event updates the latest known state of one factor. Providers with revisions
    must first reconstruct that state; publishing an old observation as current is invalid.
    The manifest records upstream verification, not a verification performed here.
    Returns the events and the manifest they were opened with.
    """
    dest = root / spec["path"]
    lf, meta = store.open(dest)
    if meta.get("known_at_verified") is not True:
        raise ValueError(
            "macro source requires verified known-at timestamps; "
            "legacy date/value cache is exploratory"
        )
    if meta.get("units") != UNITS or not meta.get("availability_method") or not meta["source"]:
        raise ValueError(
            "macro manifest needs source, availability_method and the documented factor units"
        )
    # The file is produced outside the pipeline, so check it still matches its manifest.
    if hashlib.sha256((dest / meta["file"]).read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("macro source SHA-256 differs from its manifest")
    df = lf.collect()
    if set(df.columns) != {"timestamp", "factor", "value"}:
        raise ValueError("macro source must contain timestamp, factor, value")
    dtype = df.schema["timestamp"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
        raise ValueError("macro timestamps must be tz-aware (an explicit UTC offset)")
    df = df.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.cast_time_unit("us"),
        pl.col("value").cast(pl.Float64),
    )
    if df.is_empty() or df.select(pl.struct("timestamp", "factor").is_duplicated().any()).item():
        raise ValueError("empty macro source or duplicate timestamp/factor")
    if set(df["factor"]) != set(FACTORS) or not np.isfinite(df["value"].to_numpy()).all():
        raise ValueError("macro source must contain all five factors with finite values")
    return df.sort("timestamp", "factor"), meta
