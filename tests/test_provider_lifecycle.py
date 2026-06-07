from __future__ import annotations

import threading
from pathlib import Path

import pytest
import yaml


@pytest.fixture(autouse=True)
def _immediate_background_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run provider background retains inline so SQLite stays on one thread."""

    class _ImmediateThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None) -> None:
            self._target = target

        def start(self) -> None:
            if self._target:
                self._target()

        def join(self, timeout=None) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(threading, "Thread", _ImmediateThread)


def _make_provider(hermes_home: Path, monkeypatch: pytest.MonkeyPatch, **provider_yaml: object):
    from fake_embedder import FakeEmbedder
    from turbomem import TurbomemMemoryProvider

    monkeypatch.setattr("turbomem.Embedder", FakeEmbedder)

    turbomem_dir = hermes_home / "turbomem"
    turbomem_dir.mkdir(parents=True, exist_ok=True)
    if provider_yaml:
        (turbomem_dir / "provider.yaml").write_text(
            yaml.safe_dump(provider_yaml, sort_keys=False),
            encoding="utf-8",
        )

    provider = TurbomemMemoryProvider()
    provider.initialize("sess-42", hermes_home=str(hermes_home), agent_context="primary")
    return provider


class TestPrefetchExclusions:
    def test_prefetch_excludes_conversation_and_compression(
        self, hermes_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _make_provider(hermes_home, monkeypatch)
        store = provider._store
        assert store is not None

        store.remember(text="Durable auth decision", category="general", tags=["topic:auth"])
        store.remember(text="User: hi", category="conversation", tags=["conversation"])
        store.remember(text="User: old context", category="compression", tags=["compression"])

        injected = provider.prefetch("auth decisions")
        assert "auth decision" in injected
        assert "hi" not in injected
        assert "old context" not in injected


class TestSyncTurn:
    def test_retain_every_n_turns(self, hermes_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _make_provider(
            hermes_home,
            monkeypatch,
            auto_retain_turns="true",
            retain_every_n_turns=2,
        )
        store = provider._store
        assert store is not None

        provider.sync_turn("one", "two", session_id="sess-42")
        provider.sync_turn("three", "four", session_id="sess-42")
        if provider._sync_thread and provider._sync_thread.is_alive():
            provider._sync_thread.join(timeout=3.0)

        convos = store.list_experiences(category="conversation")
        assert "three" in convos
        assert "one" not in convos

    def test_session_end_retains_summary(self, hermes_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _make_provider(
            hermes_home,
            monkeypatch,
            retain_every_n_turns=99,
        )
        store = provider._store
        assert store is not None

        provider.on_session_end(
            [
                {"role": "user", "content": "start task"},
                {"role": "assistant", "content": "acknowledged"},
            ]
        )
        listed = store.list_experiences(tags=["session-end"])
        assert "start task" in listed or "acknowledged" in listed


class TestCompressionRetain:
    def test_on_pre_compress_creates_compression_tag(
        self, hermes_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _make_provider(hermes_home, monkeypatch)
        store = provider._store
        assert store is not None

        provider.on_pre_compress(
            [
                {"role": "user", "content": "long context chunk"},
                {"role": "assistant", "content": "long reply chunk"},
            ]
        )
        if provider._compress_thread and provider._compress_thread.is_alive():
            provider._compress_thread.join(timeout=3.0)

        listed = store.list_experiences(tags=["compression"])
        assert "long context chunk" in listed or "long reply chunk" in listed


class TestMemoryWriteMirror:
    def test_on_memory_write_add_searchable(
        self, hermes_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _make_provider(hermes_home, monkeypatch)
        store = provider._store
        assert store is not None

        provider.on_memory_write("add", "user", "Always run ruff before commit")
        recalled = store.recall(query="ruff before commit", types=["experience"])
        assert "ruff" in recalled
