import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from hermes_turbomem.store import NO_HIT_HINTS


@pytest.fixture()
def repo_copy(sample_repo: Path) -> Generator[Path, None, None]:
    """Create a temporary copy of the sample repo so file deletion tests are isolated."""
    tmp = Path(tempfile.mkdtemp(prefix="turbomem_repo_"))
    for item in sample_repo.iterdir():
        if item.name in (".git", ".turbomem"):
            continue
        if item.is_dir():
            shutil.copytree(item, tmp / item.name)
        else:
            shutil.copy2(item, tmp / item.name)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


class TestHealthCheck:
    def test_health_check_no_projects(self, store):
        result = store.health_check()
        assert result == "No projects indexed; nothing to check."

    def test_health_check_clean_after_index(self, store, repo_copy):
        store.index_project(str(repo_copy))
        result = store.health_check()
        assert result == "Index health check complete: no stale entries found."

    def test_health_check_removes_stale_entries(self, store, repo_copy):
        store.index_project(str(repo_copy))
        deleted = repo_copy / "string_utils.py"
        deleted.unlink()
        result = store.health_check()
        assert "stale Code Entries" in result
        assert "Removed" in result

    def test_health_check_removes_orphaned_hashes(self, store, repo_copy):
        store.index_project(str(repo_copy))
        deleted = repo_copy / "math_utils.py"
        deleted.unlink()
        result = store.health_check()
        assert "stale Code Entries" in result or "orphaned file hashes" in result

    def test_turbovec_ids_consistent_after_cleanup(self, store, repo_copy):
        store.index_project(str(repo_copy))
        deleted = repo_copy / "string_utils.py"
        deleted.unlink()
        initial_count = len(store._index)
        store.health_check()
        after_count = len(store._index)
        assert after_count < initial_count
        db_rows = store._conn.execute("SELECT id FROM entries").fetchall()
        for r in db_rows:
            eid = int(r["id"])
            assert store._index.contains(eid), (
                f"DB entry {eid} has no matching vector in index"
            )

    def test_health_check_with_project_id_filter(self, store, repo_copy):
        store.index_project(str(repo_copy))
        from hermes_turbomem.project_id import resolve_project
        info = resolve_project(repo_copy)
        result = store.health_check(project_id=info.project_id)
        assert "no stale entries" in result

    def test_health_check_with_project_path_filter(self, store, repo_copy):
        store.index_project(str(repo_copy))
        result = store.health_check(project_path=str(repo_copy))
        assert "no stale entries" in result


class TestNoHitHints:
    def test_recall_empty_index(self, store):
        result = store.recall(query="something")
        assert "No matching memories found" in result
        assert NO_HIT_HINTS in result

    def test_recall_empty_query(self, store):
        result = store.recall(query="")
        assert "Query is empty" in result or "empty" in result.lower()

    def test_recall_no_match_with_filter(self, store, repo_copy):
        store.index_project(str(repo_copy))
        result = store.recall(query="anything", project_id="git:no/such/project")
        assert "No entries match" in result
        assert NO_HIT_HINTS in result

    def test_peek_empty_index(self, store):
        result = store.code_peek(query="something")
        assert "No matching Code Entries" in result
        assert NO_HIT_HINTS in result

    def test_peek_empty_query(self, store):
        result = store.code_peek(query="")
        assert "Query is empty" in result or "empty" in result.lower()

    def test_peek_no_match_with_filter(self, store, repo_copy):
        store.index_project(str(repo_copy))
        result = store.code_peek(query="anything", project_id="git:no/such/project")
        assert "No matching Code Entries" in result
        assert NO_HIT_HINTS in result

    def test_recall_does_not_auto_run_health_check(self, store, repo_copy):
        """Verify recall does not silently trigger health repair."""
        store.index_project(str(repo_copy))
        initial_entry_count = store._conn.execute(
            "SELECT COUNT(*) FROM entries"
        ).fetchone()[0]
        store.recall(query="something not there", project_id="git:no/such/project")
        after_count = store._conn.execute(
            "SELECT COUNT(*) FROM entries"
        ).fetchone()[0]
        assert after_count == initial_entry_count, (
            "Recall should not delete any entries"
        )


class TestCallGraphCleanup:
    def test_call_graph_edges_stored_during_index(self, store, repo_copy):
        store.index_project(str(repo_copy))
        edges = store._conn.execute(
            "SELECT COUNT(*) FROM call_graph"
        ).fetchone()[0]
        assert edges > 0, "Call graph edges should be created during indexing"

    def test_health_check_removes_orphan_call_graph_edges(
        self, store, repo_copy
    ):
        store.index_project(str(repo_copy))
        edges_before = store._conn.execute(
            "SELECT COUNT(*) FROM call_graph"
        ).fetchone()[0]
        assert edges_before > 0

        deleted = repo_copy / "string_utils.py"
        deleted.unlink()
        result = store.health_check()
        assert "stale Code Entries" in result
        edges_after = store._conn.execute(
            "SELECT COUNT(*) FROM call_graph"
        ).fetchone()[0]
        assert edges_after < edges_before, (
            "Call graph edges referencing stale entries should be reduced"
        )

    def test_absent_project_cleans_up_call_graph(self, store, repo_copy):
        store.index_project(str(repo_copy))
        from hermes_turbomem.project_id import resolve_project
        info = resolve_project(repo_copy)
        edges_before = store._conn.execute(
            "SELECT COUNT(*) FROM call_graph WHERE project_id = ?",
            (info.project_id,),
        ).fetchone()[0]
        assert edges_before > 0

        shutil.rmtree(repo_copy, ignore_errors=True)
        result = store.health_check()
        assert "absent project" in result
        edges_after = store._conn.execute(
            "SELECT COUNT(*) FROM call_graph WHERE project_id = ?",
            (info.project_id,),
        ).fetchone()[0]
        assert edges_after == 0, (
            "Call graph edges should be removed when project is absent"
        )
