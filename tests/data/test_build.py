"""gnns build-data: universes always, the selected graph's buildable sources, all-or-nothing."""

import json

import polars as pl
import pytest

import gnns.data.build as build
import gnns.data.massive as massive
from gnns.data.macro import load_macro
from tests.data.test_macro_build import END, SPEC, START, fake_massive  # noqa: F401


@pytest.fixture
def setup(tmp_path, monkeypatch, fake_massive):  # noqa: F811
    data, calls = fake_massive
    data["U1"] = data["R"]  # the shared universe's one ticker
    (tmp_path / "config" / "universes").mkdir(parents=True)
    (tmp_path / "config" / "universes" / "u.txt").write_text("U1\n")
    monkeypatch.setattr(build, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "git_sha", lambda: "test")
    cfg = {
        "data_root": "data",
        "data": {"universes": ["u"], "bar": "1d", "start": START, "end": END},
        "graph": {"name": "plain", "builder": "x"},
    }
    return cfg, tmp_path / "data", calls


def _with_macro(cfg: dict, **spec) -> dict:
    source = {"kind": "macro", "path": "macro", **SPEC, **spec}
    return cfg | {"graph": {"name": "macro_sensitivity", "sources": {"macro": source}}}


def test_graph_without_sources_builds_only_universes(setup):
    cfg, root, _ = setup
    assert build.build(cfg) == [root / "u" / "1d"]
    assert not (root / "macro").exists()


def test_selected_graph_source_is_built_and_loads(setup):
    cfg, root, _ = setup
    dests = build.build(_with_macro(cfg))
    assert dests == [root / "u" / "1d", root / "macro"]
    events, meta = load_macro(root, {"path": "macro"})
    assert meta["file"] == "macro.parquet" and meta["git_sha"] == "test"
    assert meta["start"] == str(START) and meta["end"] == str(END) and meta["bar"] == "1d"
    prices = json.loads((root / "u" / "1d" / "manifest.json").read_text())
    assert meta["built_at"] == prices["built_at"]  # one build, one timestamp
    assert events.height == 5 * 60


def test_source_without_kind_is_not_built(setup):
    cfg, root, _ = setup
    cfg = _with_macro(cfg)
    del cfg["graph"]["sources"]["macro"]["kind"]
    build.build(cfg)
    assert not (root / "macro").exists()


def test_unknown_kind_fails_before_any_download(setup):
    cfg, root, calls = setup
    with pytest.raises(ValueError, match="unknown source kind"):
        build.build(_with_macro(cfg, kind="nope"))
    assert calls == [] and not root.exists()


def test_failed_source_writes_nothing(setup):
    cfg, root, _ = setup
    with pytest.raises(ValueError, match="exactly"):  # a spec missing four factors
        build.build(_with_macro(cfg, series={"rates": {"ticker": "R"}}))
    assert not root.exists()  # the universe was fetched but not written


def test_failed_universe_writes_no_source(setup, monkeypatch):
    cfg, root, _ = setup
    real = massive.daily_bars
    monkeypatch.setattr(
        massive,
        "daily_bars",
        lambda t, *a, **k: (
            real(t, *a, **k).with_columns(pl.lit(-1.0).alias("low"))
            if t == "U1"
            else real(t, *a, **k)
        ),
    )
    with pytest.raises(ValueError, match="validation failed"):
        build.build(_with_macro(cfg))
    assert not root.exists()
