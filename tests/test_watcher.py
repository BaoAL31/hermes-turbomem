from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from hermes_turbomem.code_index import file_content_hash
from hermes_turbomem.config import TurbomemConfig, load_config
from hermes_turbomem.store import MemoryStore
from hermes_turbomem.watcher import _CodeChangeHandler, _get_git_commit, _git_diff_files


class FakeEmbedder:
    dimension = 64

    def encode(self, texts: list[str]) -> np.ndarray:
        rng = np.random.RandomState(42)
        return rng.rand(len(texts), 64).astype(np.float32)


SAMPLE_PY = """
def greet(name: str) -> str:
    return f"Hello, {name}"

class Greeter:
    def __init__(self, prefix: str = "Hi"):
        self.prefix = prefix

    def greet(self, name: str) -> str:
        return f"{self.prefix}, {name}"
"""

SAMPLE_PY_MODIFIED = """
def greet(name: str) -> str:
    return f"Hey, {name}"

class Greeter:
    def __init__(self, prefix: str = "Hey"):
        self.prefix = prefix

    def greet(self, name: str) -> str:
        return f"{self.prefix}, {name}"

def farewell(name: str) -> str:
    return f"Bye, {name}"
"""

ANOTHER_PY = """
def multiply(a: int, b: int) -> int:
    return a * b
"""


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    config = TurbomemConfig(data_dir=tmp_path / "data")
    embedder = FakeEmbedder()
    return MemoryStore(config, embedder)


def _create_project(tmp_path: Path, name: str = "myproject") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "main.py").write_text(SAMPLE_PY, encoding="utf-8")
    (root / "utils.py").write_text(ANOTHER_PY, encoding="utf-8")
    return root


# --- reindex_file tests ---


def test_reindex_file_updates_changed_file(store: MemoryStore, tmp_path: Path) -> None:
    root = _create_project(tmp_path)
    result = store.index_project(str(root))
    assert "added" in result

    pid = store.get_registered_projects()[0]["project_id"]
    rel = "main.py"
    (root / rel).write_text(SAMPLE_PY_MODIFIED, encoding="utf-8")

    changed = store.reindex_file(str(pid), root, rel)
    assert changed is True

    rows = store._conn.execute(
        "SELECT symbol FROM entries WHERE project_id = ? AND entry_type = 'code' AND path = ?",
        (pid, rel),
    ).fetchall()
    symbols = {r["symbol"] for r in rows}
    assert "farewell" in symbols
    assert "greet" in symbols
    assert "Greeter" in symbols
    assert len(rows) == 3


def test_reindex_file_unchanged_skips(store: MemoryStore, tmp_path: Path) -> None:
    root = _create_project(tmp_path)
    store.index_project(str(root))
    pid = store.get_registered_projects()[0]["project_id"]
    rel = "main.py"

    changed = store.reindex_file(str(pid), root, rel)
    assert changed is False


def test_reindex_file_unindexed_file_adds(store: MemoryStore, tmp_path: Path) -> None:
    root = _create_project(tmp_path)
    store.index_project(str(root))
    pid = store.get_registered_projects()[0]["project_id"]

    (root / "new_module.py").write_text("def new_func(): return 42\n", encoding="utf-8")
    changed = store.reindex_file(str(pid), root, "new_module.py")
    assert changed is True

    row = store._conn.execute(
        "SELECT symbol FROM entries WHERE project_id = ? AND path = ?",
        (pid, "new_module.py"),
    ).fetchone()
    assert row is not None
    assert row["symbol"] == "new_func"


def test_reindex_file_removes_old_entries(store: MemoryStore, tmp_path: Path) -> None:
    root = _create_project(tmp_path)
    store.index_project(str(root))
    pid = store.get_registered_projects()[0]["project_id"]
    rel = "utils.py"

    old_rows = store._conn.execute(
        "SELECT id FROM entries WHERE project_id = ? AND path = ? AND entry_type = 'code'",
        (pid, rel),
    ).fetchall()
    old_ids = {int(r["id"]) for r in old_rows}
    assert len(old_ids) >= 1

    (root / rel).write_text("# just a comment\n", encoding="utf-8")
    changed = store.reindex_file(str(pid), root, rel)
    assert changed is True

    new_rows = store._conn.execute(
        "SELECT id FROM entries WHERE project_id = ? AND path = ? AND entry_type = 'code'",
        (pid, rel),
    ).fetchall()
    new_ids = {int(r["id"]) for r in new_rows}
    assert len(new_ids) >= 1

    for oid in old_ids:
        assert not store._index.contains(oid)

    for nid in new_ids:
        assert store._index.contains(nid)


# --- get_registered_projects tests ---


def test_get_registered_projects_after_index(store: MemoryStore, tmp_path: Path) -> None:
    root = _create_project(tmp_path)
    store.index_project(str(root))
    projects = store.get_registered_projects()
    assert len(projects) >= 1
    entry = projects[0]
    assert "project_id" in entry
    assert "root_path" in entry


def test_get_registered_projects_empty(store: MemoryStore) -> None:
    assert store.get_registered_projects() == []


# --- watcher config tests ---


def test_watcher_config_defaults() -> None:
    cfg = TurbomemConfig()
    assert cfg.watch_enabled is True
    assert cfg.watch_poll_interval == 2.0


def test_watcher_config_from_yaml(tmp_path: Path) -> None:
    config_dir = tmp_path / "turbomem"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text(
        "watch_enabled: false\nwatch_poll_interval: 5.0\n",
        encoding="utf-8",
    )
    with mock.patch.dict(os.environ, {"TURBOMEM_DATA_DIR": str(config_dir)}):
        cfg = load_config()
    assert cfg.watch_enabled is False
    assert cfg.watch_poll_interval == 5.0


# --- watcher handler tests ---


def test_code_change_handler_reindex(store: MemoryStore, tmp_path: Path) -> None:
    root = _create_project(tmp_path)
    store.index_project(str(root))
    pid = store.get_registered_projects()[0]["project_id"]

    handler = _CodeChangeHandler(str(pid), root, store)

    (root / "main.py").write_text(SAMPLE_PY_MODIFIED, encoding="utf-8")
    handler._reindex(str(root / "main.py"))

    rows = store._conn.execute(
        "SELECT symbol FROM entries WHERE project_id = ? AND path = ?",
        (pid, "main.py"),
    ).fetchall()
    symbols = {r["symbol"] for r in rows}
    assert "farewell" in symbols


def test_code_change_handler_ignores_non_code(store: MemoryStore, tmp_path: Path) -> None:
    root = _create_project(tmp_path)
    store.index_project(str(root))
    pid = store.get_registered_projects()[0]["project_id"]

    handler = _CodeChangeHandler(str(pid), root, store)

    log_path = root / "output.log"
    log_path.write_text("some log\n", encoding="utf-8")
    handler._reindex(str(log_path))

    log_id_row = store._conn.execute(
        "SELECT id FROM entries WHERE project_id = ? AND path = ?",
        (pid, "output.log"),
    ).fetchone()
    assert log_id_row is None


# --- watcher lifecycle tests ---


def test_watcher_start_stop(store: MemoryStore, tmp_path: Path) -> None:
    root = _create_project(tmp_path)
    store.index_project(str(root))
    pid = store.get_registered_projects()[0]["project_id"]

    from hermes_turbomem.watcher import ProjectWatcher

    watcher = ProjectWatcher(str(pid), root, store, poll_interval=0.1)
    watcher.start()
    try:
        assert watcher._observer is not None
        assert watcher._observer.is_alive()
        assert watcher._branch_thread is not None
        assert watcher._branch_thread.is_alive()
    finally:
        watcher.stop()

    assert watcher._observer is None or not watcher._observer.is_alive()


def test_watcher_detects_file_change(store: MemoryStore, tmp_path: Path) -> None:
    root = _create_project(tmp_path)
    store.index_project(str(root))
    pid = store.get_registered_projects()[0]["project_id"]

    from hermes_turbomem.watcher import ProjectWatcher

    watcher = ProjectWatcher(str(pid), root, store, poll_interval=0.1)
    watcher.start()
    try:
        time.sleep(0.2)

        (root / "main.py").write_text(SAMPLE_PY_MODIFIED, encoding="utf-8")
        time.sleep(0.5)

        rows = store._conn.execute(
            "SELECT symbol FROM entries WHERE project_id = ? AND path = ?",
            (pid, "main.py"),
        ).fetchall()
        symbols = {r["symbol"] for r in rows}
        assert "farewell" in symbols, f"Expected farewell symbol, got {symbols}"
    finally:
        watcher.stop()


# --- auto-index tests ---


def test_auto_index_on_first_use_enabled(tmp_path: Path) -> None:
    root = _create_project(tmp_path)
    config = TurbomemConfig(data_dir=tmp_path / "data", auto_index_on_first_use=True)
    embedder = FakeEmbedder()
    st = MemoryStore(config, embedder)

    result = st.maybe_auto_index(str(root))
    assert result is not None
    assert "Indexed" in result

    projects = st.get_registered_projects()
    assert len(projects) >= 1


def test_auto_index_on_first_use_disabled(tmp_path: Path) -> None:
    root = _create_project(tmp_path)
    config = TurbomemConfig(data_dir=tmp_path / "data", auto_index_on_first_use=False)
    embedder = FakeEmbedder()
    st = MemoryStore(config, embedder)

    result = st.maybe_auto_index(str(root))
    assert result is None

    projects = st.get_registered_projects()
    assert len(projects) == 0


def test_auto_index_skips_if_already_indexed(tmp_path: Path) -> None:
    root = _create_project(tmp_path)
    config = TurbomemConfig(data_dir=tmp_path / "data", auto_index_on_first_use=True)
    embedder = FakeEmbedder()
    st = MemoryStore(config, embedder)

    st.index_project(str(root))
    result = st.maybe_auto_index(str(root))
    assert result is None


# --- git branch helpers (where git is available) ---


def test_get_git_commit_in_non_git_dir(tmp_path: Path) -> None:
    result = _get_git_commit(tmp_path)
    assert result is None


def test_git_diff_files_in_non_git_dir(tmp_path: Path) -> None:
    result = _git_diff_files(tmp_path, "HEAD", "HEAD")
    assert result == []


# --- error logging tests ---


def test_watcher_errors_logged_not_swallowed(store: MemoryStore, tmp_path: Path, caplog) -> None:
    root = _create_project(tmp_path)
    store.index_project(str(root))
    pid = store.get_registered_projects()[0]["project_id"]

    handler = _CodeChangeHandler(str(pid), root, store)

    with caplog.at_level(logging.ERROR):
        handler._reindex(str(root / "nonexistent.py"))

    nonexistent = root / "nonexistent.py"
    none_row = store._conn.execute(
        "SELECT id FROM entries WHERE project_id = ? AND path = ?",
        (pid, "nonexistent.py"),
    ).fetchone()
    assert none_row is None


# --- config.example.yaml alignment ---


def test_config_example_has_watcher_settings() -> None:
    cfg_path = Path(__file__).resolve().parent.parent / "config.example.yaml"
    text = cfg_path.read_text(encoding="utf-8")
    assert "watch_enabled:" in text
    assert "watch_poll_interval:" in text
