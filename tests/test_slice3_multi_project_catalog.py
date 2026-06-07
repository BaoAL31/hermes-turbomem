from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_turbomem.project_id import resolve_project
from hermes_turbomem.store import MemoryStore

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_repo_b() -> Path:
    repo = FIXTURES / "sample_repo_b"
    if not (repo / ".git").exists():
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=fixture@test", "-c", "user.name=fixture", "commit", "-m", "fixture"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
    return repo


def test_two_projects_indexed_in_catalog(
    store: MemoryStore, sample_repo: Path, sample_repo_b: Path
) -> None:
    store.index_codebase(str(sample_repo))
    store.index_codebase(str(sample_repo_b))
    listing = store.list_code_projects()
    assert str(sample_repo.resolve()) in listing
    assert str(sample_repo_b.resolve()) in listing


def test_code_recall_cross_project_without_filter(
    store: MemoryStore, sample_repo: Path, sample_repo_b: Path
) -> None:
    store.index_codebase(str(sample_repo))
    store.index_codebase(str(sample_repo_b))
    math_hit = store.code_recall("add two numbers")
    assert "math_utils.py" in math_hit or "add" in math_hit
    assert str(sample_repo.resolve()) in math_hit or "sample_repo" in math_hit
    api_hit = store.code_recall("api_handler endpoint status")
    assert "api_handler.py" in api_hit or "api_handler" in api_hit
    assert str(sample_repo_b.resolve()) in api_hit or "sample_repo_b" in api_hit


def test_code_recall_project_id_filter(
    store: MemoryStore, sample_repo: Path, sample_repo_b: Path
) -> None:
    store.index_codebase(str(sample_repo))
    store.index_codebase(str(sample_repo_b))
    repo_b_id = resolve_project(sample_repo_b).project_id
    result = store.code_recall("route HTTP request", project_id=repo_b_id)
    assert "web_handler.py" in result or "route_path" in result
    assert "math_utils.py" not in result


def test_code_recall_project_path_filter(
    store: MemoryStore, sample_repo: Path, sample_repo_b: Path
) -> None:
    store.index_codebase(str(sample_repo))
    store.index_codebase(str(sample_repo_b))
    result = store.code_recall("multiply numbers", project_path=str(sample_repo))
    assert "math_utils.py" in result or "multiply" in result
    assert "web_handler.py" not in result


def test_code_recall_outside_project_cwd(
    store: MemoryStore, sample_repo: Path, sample_repo_b: Path, tmp_path: Path
) -> None:
    store.index_codebase(str(sample_repo))
    store.index_codebase(str(sample_repo_b))
    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    old = os.getcwd()
    try:
        os.chdir(workdir)
        result = store.code_recall("api handler endpoint")
        assert "api_handler" in result or "api_handler.py" in result
    finally:
        os.chdir(old)
