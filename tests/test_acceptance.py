from __future__ import annotations

from pathlib import Path

import pytest

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


def test_empty_recall_before_index(store: MemoryStore) -> None:
    result = store.code_recall("anything")
    assert "No projects indexed yet" in result


def test_recall_with_empty_query(store: MemoryStore, sample_repo: Path) -> None:
    store.index_codebase(str(sample_repo))
    result = store.code_recall("")
    assert "empty" in result.lower()


def test_multi_project_recall_returns_both(
    store: MemoryStore, sample_repo: Path, utils_project: Path
) -> None:
    store.index_codebase(str(sample_repo))
    store.index_codebase(str(utils_project))
    result = store.code_recall("function")
    assert result.count("[code | ") >= 2


def test_multi_project_recall_project_id_filter(
    store: MemoryStore, sample_repo: Path, utils_project: Path
) -> None:
    r1 = store.index_codebase(str(sample_repo))
    r2 = store.index_codebase(str(utils_project))

    proj_id_a = r1.split()[2]
    proj_id_b = r2.split()[2]

    filtered = store.code_recall("function", project_id=proj_id_a)
    assert "matching" not in filtered.lower() or "No matching" not in filtered
    assert proj_id_a in filtered


def test_multi_project_recall_project_path_filter(
    store: MemoryStore, sample_repo: Path, utils_project: Path
) -> None:
    r1 = store.index_codebase(str(sample_repo))
    store.index_codebase(str(utils_project))

    proj_id_a = r1.split()[2]

    filtered = store.code_recall("function", project_path=str(sample_repo))
    assert "matching" not in filtered.lower() or "No matching" not in filtered
    assert proj_id_a in filtered


def test_multi_project_recall_results_label_project(
    store: MemoryStore, sample_repo: Path, utils_project: Path
) -> None:
    store.index_codebase(str(sample_repo))
    store.index_codebase(str(utils_project))
    result = store.code_recall("multiply")
    lines = result.split("\n")
    code_lines = [l for l in lines if l.startswith("[code | ")]
    assert len(code_lines) > 0
    for cl in code_lines:
        assert cl.startswith("[code | ")
        assert " | score" in cl
        parts = cl.split(" | ")
        assert len(parts) >= 2
        project_label = parts[1]
        assert len(project_label) > 0


def test_multi_project_recall_empty_filter_no_match(
    store: MemoryStore, sample_repo: Path, utils_project: Path
) -> None:
    store.index_codebase(str(sample_repo))
    store.index_codebase(str(utils_project))
    result = store.code_recall("function", project_id="git:nonexistent/repo")
    assert "No matching" in result


def test_recall_from_outside_both_roots(
    store: MemoryStore, sample_repo: Path, utils_project: Path
) -> None:
    store.index_codebase(str(sample_repo))
    store.index_codebase(str(utils_project))
    result = store.code_recall("Calculator")
    assert "Calculator" in result or "AdvancedCalculator" in result
