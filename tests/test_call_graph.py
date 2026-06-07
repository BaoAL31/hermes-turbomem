from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hermes_turbomem.call_graph import extract_edges, SUPPORTED_LANGUAGES
from hermes_turbomem.code_index import TS_LANGUAGE_MAP
from hermes_turbomem.config import TurbomemConfig
from hermes_turbomem.store import MemoryStore

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "call_graph"


class FakeEmbedder:
    dimension = 768

    def __init__(self, model_name: str = "fake") -> None:
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 768), dtype=np.float32)


# ΓöÇΓöÇ extract_edges unit tests ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


@pytest.mark.parametrize(
    "filename,expected_edges",
    [
        ("call_chain.py", [("run", "helper")]),
        ("call_chain.js", [("run", "helper")]),
        ("call_chain.ts", [("run", "helper")]),
        ("call_chain.go", [("run", "helper")]),
        ("call_chain.rs", [("run", "helper")]),
    ],
)
def test_extract_edges(filename: str, expected_edges: list[tuple[str, str]]) -> None:
    fixture = FIXTURE_DIR / filename
    source = fixture.read_text(encoding="utf-8")
    edges = extract_edges(fixture, source)
    assert edges is not None
    assert len(edges) == len(expected_edges)
    for edge, (caller, callee) in zip(edges, expected_edges, strict=False):
        assert edge.caller_symbol == caller
        assert edge.callee == callee


def test_extract_edges_unsupported() -> None:
    path = Path("dummy.java")
    source = "public class Foo { void bar() { baz(); } }"
    result = extract_edges(path, source)
    assert result is None


def test_extract_edges_no_calls() -> None:
    fixture = FIXTURE_DIR / "call_chain.py"
    source = "def helper():\n    return 42\n"
    edges = extract_edges(fixture, source)
    assert edges is None


# ΓöÇΓöÇ Integration: MemoryStore + code_call_graph ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


def _make_store(tmpdir: Path) -> MemoryStore:
    data_dir = tmpdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = TurbomemConfig(
        data_dir=data_dir,
        auto_index_on_first_use=False,
        default_recall_limit=8,
    )
    return MemoryStore(cfg, FakeEmbedder())


def test_code_call_graph_callees(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    conn = store._conn

    conn.execute(
        "INSERT INTO projects (project_id, root_path, indexed_at) VALUES (?, ?, ?)",
        ("test:proj", "/fake/root", 1000.0),
    )
    conn.execute(
        """
        INSERT INTO entries (id, entry_type, project_id, text, path, symbol, start_line, end_line, created_at)
        VALUES (1, 'code', 'test:proj', 'def run(): helper()', 'main.py', 'run', 1, 2, 1000.0)
        """,
    )
    conn.execute(
        """
        INSERT INTO entries (id, entry_type, project_id, text, path, symbol, start_line, end_line, created_at)
        VALUES (2, 'code', 'test:proj', 'def helper(): pass', 'main.py', 'helper', 4, 5, 1000.0)
        """,
    )
    conn.execute(
        "INSERT INTO call_edges (project_id, path, caller_symbol, callee, caller_start_line, caller_end_line, callee_line) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("test:proj", "main.py", "run", "helper", 1, 2, 1),
    )
    conn.commit()

    result = store.code_call_graph("run", direction="callees", project_id="test:proj")
    assert "helper" in result
    conn.close()


def test_code_call_graph_callers(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    conn = store._conn

    conn.execute(
        "INSERT INTO projects (project_id, root_path, indexed_at) VALUES (?, ?, ?)",
        ("test:proj", "/fake/root", 1000.0),
    )
    conn.execute(
        """
        INSERT INTO entries (id, entry_type, project_id, text, path, symbol, start_line, end_line, created_at)
        VALUES (1, 'code', 'test:proj', 'def run(): helper()', 'main.py', 'run', 1, 2, 1000.0)
        """,
    )
    conn.execute(
        """
        INSERT INTO entries (id, entry_type, project_id, text, path, symbol, start_line, end_line, created_at)
        VALUES (2, 'code', 'test:proj', 'def helper(): pass', 'main.py', 'helper', 4, 5, 1000.0)
        """,
    )
    conn.execute(
        "INSERT INTO call_edges (project_id, path, caller_symbol, callee, caller_start_line, caller_end_line, callee_line) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("test:proj", "main.py", "run", "helper", 1, 2, 1),
    )
    conn.commit()

    result = store.code_call_graph("helper", direction="callers", project_id="test:proj")
    assert "run" in result
    conn.close()


def test_code_call_graph_no_results(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store._conn.execute(
        "INSERT INTO projects (project_id, root_path, indexed_at) VALUES (?, ?, ?)",
        ("test:proj", "/fake/root", 1000.0),
    )
    store._conn.commit()

    result = store.code_call_graph("ghost", direction="callees", project_id="test:proj")
    assert "No callees" in result
    store._conn.close()


def test_code_call_graph_unsupported_language(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    conn = store._conn

    conn.execute(
        "INSERT INTO projects (project_id, root_path, indexed_at) VALUES (?, ?, ?)",
        ("test:proj", "/fake/root", 1000.0),
    )
    conn.execute(
        """
        INSERT INTO entries (id, entry_type, project_id, text, path, symbol, start_line, end_line, created_at)
        VALUES (1, 'code', 'test:proj', 'class Foo { void bar() {} }', 'Dummy.java', 'Foo', 1, 3, 1000.0)
        """,
    )
    conn.commit()

    result = store.code_call_graph("Foo", direction="callees", project_id="test:proj")
    assert "not supported" in result.lower()
    assert "java" in result.lower()
    conn.close()


def test_code_call_graph_invalid_direction(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    result = store.code_call_graph("foo", direction="sideways")
    assert "Invalid direction" in result
    store._conn.close()


# ΓöÇΓöÇ Full end-to-end: index_project ΓåÆ code_call_graph ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


@pytest.fixture
def indexed_project(tmp_path: Path) -> MemoryStore:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text(
        "def helper() -> int:\n"
        "    return 42\n"
        "\n"
        "def run() -> int:\n"
        "    val = helper()\n"
        "    return val\n"
    )
    (repo / "main.js").write_text(
        "function helper() {\n"
        "    return 42;\n"
        "}\n"
        "\n"
        "function run() {\n"
        "    return helper();\n"
        "}\n"
    )
    store = _make_store(tmp_path / "store")
    store.index_project(str(repo))
    return store


def test_full_index_and_call_graph_callees(indexed_project: MemoryStore) -> None:
    store = indexed_project
    result = store.code_call_graph("run", direction="callees")
    assert "helper" in result
    store._conn.close()


def test_full_index_and_call_graph_callers(indexed_project: MemoryStore) -> None:
    store = indexed_project
    result = store.code_call_graph("helper", direction="callers")
    assert "run" in result
    store._conn.close()


def test_full_index_unsupported_language_message(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Main.java").write_text(
        "public class Main {\n"
        "    void run() { helper(); }\n"
        "    void helper() {}\n"
        "}\n"
    )
    store = _make_store(tmp_path / "store")
    store.index_project(str(repo))

    result = store.code_call_graph("Main", direction="callees")
    assert "not supported" in result.lower()
    store._conn.close()


def test_incremental_reindex_preserves_call_graph(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pyfile = repo / "main.py"
    pyfile.write_text(
        "def helper() -> int:\n"
        "    return 42\n"
        "\n"
        "def run() -> int:\n"
        "    val = helper()\n"
        "    return val\n"
    )
    store = _make_store(tmp_path / "store")
    store.index_project(str(repo))

    result_before = store.code_call_graph("run", direction="callees")
    assert "helper" in result_before

    store.index_project(str(repo))

    result_after = store.code_call_graph("run", direction="callees")
    assert "helper" in result_after
    store._conn.close()


def test_incremental_reindex_unchanged_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pyfile = repo / "main.py"
    pyfile.write_text(
        "def helper() -> int:\n"
        "    return 42\n"
        "\n"
        "def run() -> int:\n"
        "    val = helper()\n"
        "    return val\n"
    )
    store = _make_store(tmp_path / "store")
    store.index_project(str(repo))

    count_before = store._conn.execute(
        "SELECT COUNT(*) FROM call_edges"
    ).fetchone()[0]

    store.index_project(str(repo))

    count_after = store._conn.execute(
        "SELECT COUNT(*) FROM call_edges"
    ).fetchone()[0]

    assert count_before == count_after
    result = store.code_call_graph("run", direction="callees")
    assert "helper" in result
    store._conn.close()
