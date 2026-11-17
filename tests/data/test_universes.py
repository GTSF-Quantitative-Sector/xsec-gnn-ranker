"""Universe definitions: config/universes/<name>.yaml, validated by kind."""

import pytest

from gnns.config import CONFIG_DIR
from gnns.data import universes


def _write(tmp_path, name: str, text: str):
    (tmp_path / "universes").mkdir(exist_ok=True)
    (tmp_path / "universes" / f"{name}.yaml").write_text(text)
    return tmp_path


def test_tracked_universes_are_valid_lists():
    for name in ("sector_etfs", "benchmarks"):
        s = universes.spec(name, CONFIG_DIR)
        assert s["kind"] == "list" and s["tickers"]
    assert {"SPY", "BIL"} <= set(universes.spec("benchmarks", CONFIG_DIR)["tickers"])


def test_list_universe(tmp_path):
    d = _write(tmp_path, "u", "kind: list\ntickers: [A, B]\n")
    assert universes.spec("u", d)["tickers"] == ["A", "B"]


def test_membership_universe_passes_through(tmp_path):
    d = _write(tmp_path, "m", "kind: membership\nsource: datastream\nlist_code: 4408\n")
    assert universes.spec("m", d)["list_code"] == 4408


@pytest.mark.parametrize(
    "text, match",
    [
        ("tickers: [A]\n", "kind must be one of"),
        ("kind: basket\ntickers: [A]\n", "kind must be one of"),
        ("kind: list\n", "non-empty list of tickers"),
        ("kind: list\ntickers: []\n", "non-empty list of tickers"),
        ("kind: list\ntickers: [A, A]\n", r"duplicate tickers \['A'\]"),
    ],
)
def test_invalid_definitions_raise(tmp_path, text, match):
    d = _write(tmp_path, "bad", text)
    with pytest.raises(ValueError, match=match):
        universes.spec("bad", d)


def test_missing_definition_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="no definition at"):
        universes.spec("nope", tmp_path)
