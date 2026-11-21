"""WRDS helpers that run without a connection: row conversion and the extract cache."""

from decimal import Decimal

import polars as pl
import pytest

from gnns.data import store, wrds
from tests.data.test_store import MANIFEST


class FakeCursor:
    def __init__(self, names, rows):
        self.description = [type("D", (), {"name": n})() for n in names]
        self._rows = rows

    def execute(self, sql, params=None):
        self.sql, self.params = sql, params

    def fetchall(self):
        return self._rows


def test_query_turns_numeric_columns_into_floats():
    cur = FakeCursor(["infocode", "ticker"], [(Decimal("72405"), "AAPL"), (Decimal("1.5"), "X")])
    df = wrds.query(cur, "SELECT 1", {"a": 1})
    assert df.schema == {"infocode": pl.Float64, "ticker": pl.String}
    assert df["infocode"].to_list() == [72405.0, 1.5]
    assert cur.params == {"a": 1}


def test_load_names_fetch_wrds_when_an_extract_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="gnns fetch-wrds"):
        wrds.load(tmp_path, "ds_tickers")
    store.write(pl.DataFrame({"a": [1]}), tmp_path / "wrds" / "x", MANIFEST, file="x.parquet")
    df, manifest = wrds.load(tmp_path, "x")
    assert df["a"].to_list() == [1] and manifest["file"] == "x.parquet"


def test_connect_without_a_username_says_where_to_set_it(monkeypatch):
    monkeypatch.setattr(wrds, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    with pytest.raises(RuntimeError, match="WRDS_USERNAME"):
        wrds.connect()
