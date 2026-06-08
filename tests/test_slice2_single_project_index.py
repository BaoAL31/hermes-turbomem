from __future__ import annotations

from pathlib import Path

from hermes_turbomem.code_index import iter_indexable_files
from hermes_turbomem.store import MemoryStore


def test_index_codebase_indexes_fixture(store: MemoryStore, sample_repo: Path) -> None:
    result = store.index_codebase(str(sample_repo))
    assert "Indexed project" in result
    assert "code entries added" in result


def test_list_code_projects_shows_indexed_project(store: MemoryStore, sample_repo: Path) -> None:
    store.index_codebase(str(sample_repo))
    listing = store.list_code_projects()
    assert str(sample_repo.resolve()) in listing


def test_code_recall_returns_ranked_entries(store: MemoryStore, sample_repo: Path) -> None:
    store.index_codebase(str(sample_repo))
    result = store.code_recall("add two numbers")
    assert "score" in result
    assert "math_utils.py" in result or "add" in result


def test_reindex_unchanged_files_skips(store: MemoryStore, sample_repo: Path) -> None:
    store.index_codebase(str(sample_repo))
    result2 = store.index_codebase(str(sample_repo))
    assert "0 code entries added" in result2 or "skipped" in result2


def test_index_scope_excludes_gitignored(store: MemoryStore, sample_repo: Path) -> None:
    files = iter_indexable_files(sample_repo)
    rels = [f.relative_to(sample_repo).as_posix() for f in files]
    assert "ignored_file.py" not in rels


def test_index_scope_excludes_denied_binary(store: MemoryStore, sample_repo: Path) -> None:
    files = iter_indexable_files(sample_repo)
    rels = [f.relative_to(sample_repo).as_posix() for f in files]
    assert "denied_binary.bin" not in rels


def test_empty_code_recall_before_index(store: MemoryStore) -> None:
    result = store.code_recall("anything")
    assert "No projects indexed yet" in result
