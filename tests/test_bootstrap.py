import importlib
import os
import sys
from pathlib import Path

import pytest


def _clean_modules():
    for mod in list(sys.modules.keys()):
        if "hermes_turbomem" in mod:
            del sys.modules[mod]


def _import_server(data_dir: Path):
    _clean_modules()
    os.environ["TURBOMEM_DATA_DIR"] = str(data_dir)
    import hermes_turbomem.server as srv
    return srv


class TestListCodeProjects:
    def test_empty_catalog(self, tmp_path):
        srv = _import_server(tmp_path / "data")
        result = srv.list_code_projects()
        assert "No projects" in result


class TestIndexStatus:
    def test_reports_zero_chunks(self, tmp_path):
        srv = _import_server(tmp_path / "data")
        result = srv.index_status()
        assert "0 project(s)" in result
        assert "0 code entry" in result
        assert "not cached" in result or "cached" in result

    def test_reports_never_indexed(self, tmp_path):
        srv = _import_server(tmp_path / "data")
        result = srv.index_status()
        assert "never" in result


class TestPreloadModels:
    def test_returns_success_message(self, tmp_path, monkeypatch):
        srv = _import_server(tmp_path / "data")
        monkeypatch.setattr(srv._embedder, "preload", lambda: None)
        result = srv.preload_models()
        assert "loaded and cached" in result

    def test_handles_failure(self, tmp_path, monkeypatch):
        srv = _import_server(tmp_path / "data")
        def _fail():
            raise RuntimeError("no network")
        monkeypatch.setattr(srv._embedder, "preload", _fail)
        result = srv.preload_models()
        assert "Failed" in result


class TestConfig:
    def test_defaults(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "turbomem"
        data_dir.mkdir(parents=True)
        monkeypatch.setenv("TURBOMEM_DATA_DIR", str(data_dir))
        _clean_modules()
        from hermes_turbomem.config import load_config
        cfg = load_config()
        assert cfg.data_dir == data_dir
        assert cfg.auto_index_on_first_use is False
        assert cfg.embedding_model == "nomic-ai/nomic-embed-text-v1"
        assert cfg.bit_width == 4
        assert cfg.default_recall_limit == 8

    def test_override_from_file(self, tmp_path, monkeypatch):
        import yaml
        data_dir = tmp_path / "turbomem"
        data_dir.mkdir(parents=True)
        config_path = data_dir / "config.yaml"
        config_path.write_text(
            yaml.dump({"embedding_model": "other/model", "bit_width": 8}),
            encoding="utf-8",
        )
        monkeypatch.setenv("TURBOMEM_DATA_DIR", str(data_dir))
        _clean_modules()
        from hermes_turbomem.config import load_config
        cfg = load_config()
        assert cfg.embedding_model == "other/model"
        assert cfg.bit_width == 8
        assert cfg.default_recall_limit == 8  # kept default


class TestEmbedder:
    def test_lazy_init(self, tmp_path, monkeypatch):
        srv = _import_server(tmp_path / "data")
        assert srv._embedder.is_loaded is False

    def test_config_loads_with_env_override(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "custom"
        data_dir.mkdir(parents=True)
        monkeypatch.setenv("TURBOMEM_DATA_DIR", str(data_dir))
        _clean_modules()
        from hermes_turbomem.config import load_config
        cfg = load_config()
        assert cfg.data_dir == data_dir


class TestServerName:
    def test_fastmcp_name(self, tmp_path):
        srv = _import_server(tmp_path / "data")
        assert srv.mcp.name == "hermes-turbomem"
