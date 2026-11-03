import gnns.config as config


def test_machine_yaml_selects_construction_before_it_is_resolved(tmp_path, monkeypatch):
    (tmp_path / "graphs").mkdir()
    (tmp_path / "config.yaml").write_text("graph: shared\nnormalize: rank\n")
    (tmp_path / "machine.yaml").write_text("graph: local\nnormalize: zscore\n")
    (tmp_path / "graphs" / "shared.yaml").write_text("builder: shared_builder\n")
    (tmp_path / "graphs" / "local.yaml").write_text("builder: local_builder\n")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    cfg = config.load_config()

    assert cfg["graph"] == {"name": "local", "builder": "local_builder"}
    assert cfg["normalize"] == "zscore"
    assert cfg["data_root"] == "data"


def test_without_machine_yaml_uses_shared_construction(tmp_path, monkeypatch):
    (tmp_path / "graphs").mkdir()
    (tmp_path / "config.yaml").write_text("graph: shared\n")
    (tmp_path / "graphs" / "shared.yaml").write_text("builder: shared_builder\n")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    cfg = config.load_config()

    assert cfg["graph"] == {"name": "shared", "builder": "shared_builder"}
    assert cfg["data_root"] == "data"
