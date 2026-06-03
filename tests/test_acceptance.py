from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_turbomem.code_index import (
    DENY_PATTERNS,
    MAX_FILE_SIZE,
    iter_indexable_files,
)
from hermes_turbomem.config import TurbomemConfig
from hermes_turbomem.embedder import Embedder
from hermes_turbomem.store import MemoryStore


@pytest.fixture
def cfg(tmp_path: Path) -> TurbomemConfig:
    return TurbomemConfig(data_dir=tmp_path / ".hermes" / "turbomem")


@pytest.fixture
def embedder() -> Embedder:
    return Embedder("all-MiniLM-L6-v2")


@pytest.fixture
def store(cfg: TurbomemConfig, embedder: Embedder) -> MemoryStore:
    return MemoryStore(cfg, embedder)


def test_index_codebase_creates_turbomem_dir(store: MemoryStore, sample_repo: Path) -> None:
    result = store.index_codebase(str(sample_repo))
    assert "Indexed project" in result
    assert "code entries added" in result
    turbomem_dir = sample_repo / ".turbomem"
    assert turbomem_dir.is_dir()
    assert (turbomem_dir / "project_index.db").is_file()
    assert (turbomem_dir / "index.tvim").is_file()


def test_list_code_projects_shows_indexed_project(store: MemoryStore, sample_repo: Path) -> None:
    store.index_codebase(str(sample_repo))
    listing = store.list_code_projects()
    assert "git" in listing or "local" in listing
    assert str(sample_repo.resolve()) in listing


def test_code_recall_returns_ranked_entries(store: MemoryStore, sample_repo: Path) -> None:
    store.index_codebase(str(sample_repo))
    result = store.code_recall("add two numbers")
    assert "score" in result
    assert "add" in result
    assert "math_utils.py" in result or "Calculator" in result


def test_code_recall_returns_symbol_path_line_range(store: MemoryStore, sample_repo: Path) -> None:
    store.index_codebase(str(sample_repo))
    result = store.code_recall("multiply")
    assert "multiply" in result.lower() or "Multiply" in result
    assert "math_utils.py" in result
    assert ":" in result


def test_reindex_unchanged_files_skips(store: MemoryStore, sample_repo: Path) -> None:
    result1 = store.index_codebase(str(sample_repo))
    result2 = store.index_codebase(str(sample_repo))
    assert "0 code entries added" in result2 or "skipped" in result2
    assert "Indexed project" in result2


def test_index_scope_excludes_gitignored(store: MemoryStore, sample_repo: Path) -> None:
    files = iter_indexable_files(sample_repo)
    rels = [f.relative_to(sample_repo).as_posix() for f in files]
    assert "ignored_file.py" not in rels


def test_index_scope_excludes_denied_binary(store: MemoryStore, sample_repo: Path) -> None:
    files = iter_indexable_files(sample_repo)
    rels = [f.relative_to(sample_repo).as_posix() for f in files]
    assert "denied_binary.bin" not in rels


def test_no_remember_tool() -> None:
    import hermes_turbomem.server as server_mod
    tool_names = {
        name for name in dir(server_mod)
        if not name.startswith("_") and callable(getattr(server_mod, name, None))
    }
    assert "remember" not in tool_names, "remember tool must not be exposed in complement mode"
    assert "index_codebase" in tool_names
    assert "code_recall" in tool_names
    assert "list_code_projects" in tool_names


def test_empty_recall_before_index(store: MemoryStore) -> None:
    result = store.code_recall("anything")
    assert "No projects indexed yet" in result


def test_recall_with_empty_query(store: MemoryStore, sample_repo: Path) -> None:
    store.index_codebase(str(sample_repo))
    result = store.code_recall("")
    assert "empty" in result.lower()
