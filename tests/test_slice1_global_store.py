from __future__ import annotations

from pathlib import Path

import pytest

from hermes_turbomem.config import TurbomemConfig
from hermes_turbomem.store import MemoryStore
from tests.fake_embedder import FakeEmbedder


class TestGlobalStoreBootstrap:
    def test_initializes_sqlite_and_turbovec_paths(self, turbomem_config: TurbomemConfig) -> None:
        store = MemoryStore(turbomem_config, FakeEmbedder())
        assert turbomem_config.data_dir.is_dir()
        assert turbomem_config.db_path.is_file()
        store.remember(text="bootstrap probe")
        assert turbomem_config.index_path.is_file()


class TestExperiencePath:
    def test_retain_recall_list(self, store: MemoryStore) -> None:
        text = "Always use bcrypt for password hashing in this project."
        store.remember(text=text, category="general")
        recalled = store.recall(query=text, types=["experience"])
        assert "bcrypt" in recalled
        listed = store.list_experiences()
        assert "bcrypt" in listed
        assert "1 experience(s)" in listed


class TestTagFilter:
    def test_recall_any_tag_overlap(self, store: MemoryStore) -> None:
        store.remember(text="Auth uses JWT tokens", tags=["topic:auth"])
        store.remember(text="Deploy via GitHub Actions", tags=["topic:deploy"])
        recalled = store.recall(
            query="authentication tokens",
            types=["experience"],
            tags=["topic:auth"],
        )
        assert "JWT" in recalled
        assert "GitHub Actions" not in recalled

    def test_list_any_tag_overlap(self, store: MemoryStore) -> None:
        store.remember(text="Auth uses JWT tokens", tags=["topic:auth", "session:1"])
        store.remember(text="Deploy via GitHub Actions", tags=["topic:deploy"])
        listed = store.list_experiences(tags=["session:1", "topic:deploy"])
        assert "JWT" in listed
        assert "GitHub Actions" in listed
        assert "2 experience(s)" in listed


class TestEmptyCatalogDiagnostics:
    def test_list_code_projects_empty(self, store: MemoryStore) -> None:
        result = store.list_code_projects()
        assert "No projects indexed yet" in result
        assert "index_codebase" in result

    def test_index_status_empty(self, store: MemoryStore) -> None:
        result = store.index_status()
        assert "0 project(s)" in result
        assert "0 code entry" in result
        assert "never" in result
        assert "index_codebase" in result


class TestProviderDispatch:
    @pytest.fixture(autouse=True)
    def _fake_embedder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("turbomem.Embedder", FakeEmbedder)

    def test_memory_store_retain_recall_list(self, hermes_home: Path) -> None:
        from turbomem import TurbomemMemoryProvider

        provider = TurbomemMemoryProvider()
        provider.initialize("test-session", hermes_home=str(hermes_home), agent_context="primary")

        retain = provider.handle_tool_call(
            "memory_store",
            {"action": "retain", "text": "Prefer pytest fixtures for store tests."},
        )
        assert "Stored experience" in retain

        recalled = provider.handle_tool_call(
            "memory_store",
            {"action": "recall", "query": "pytest fixtures"},
        )
        assert "pytest fixtures" in recalled

        listed = provider.handle_tool_call("memory_store", {"action": "list"})
        assert "pytest fixtures" in listed

    def test_provider_tag_filter(self, hermes_home: Path) -> None:
        from turbomem import TurbomemMemoryProvider

        provider = TurbomemMemoryProvider()
        provider.initialize("test-session", hermes_home=str(hermes_home), agent_context="primary")

        provider.handle_tool_call(
            "memory_store",
            {"action": "retain", "text": "Use RRF for hybrid search", "tags": ["topic:search"]},
        )
        provider.handle_tool_call(
            "memory_store",
            {"action": "retain", "text": "SQLite stores metadata", "tags": ["topic:storage"]},
        )

        recalled = provider.handle_tool_call(
            "memory_store",
            {"action": "recall", "query": "hybrid search fusion", "tags": ["topic:search"]},
        )
        assert "RRF" in recalled
        assert "SQLite" not in recalled

    def test_preload_models(self, hermes_home: Path) -> None:
        from turbomem import TurbomemMemoryProvider

        provider = TurbomemMemoryProvider()
        provider.initialize("test-session", hermes_home=str(hermes_home), agent_context="primary")
        result = provider.handle_tool_call("preload_models", {})
        assert "Embedding model ready" in result
        assert "dim=" in result

    def test_list_code_projects_and_index_status(self, hermes_home: Path) -> None:
        from turbomem import TurbomemMemoryProvider

        provider = TurbomemMemoryProvider()
        provider.initialize("test-session", hermes_home=str(hermes_home), agent_context="primary")

        projects = provider.handle_tool_call("list_code_projects", {})
        assert "No projects indexed yet" in projects

        status = provider.handle_tool_call("index_status", {})
        assert "0 project(s)" in status
        assert "never" in status
