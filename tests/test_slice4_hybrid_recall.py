from __future__ import annotations

from pathlib import Path

import pytest

from hermes_turbomem.store import MemoryStore


def test_code_recall_hybrid_bm25_symbol_match(store: MemoryStore, sample_repo: Path) -> None:
    store.index_codebase(str(sample_repo))
    result = store.code_recall("subtract")
    assert "subtract" in result


def test_code_peek_omits_source_body(store: MemoryStore, sample_repo: Path) -> None:
    store.index_codebase(str(sample_repo))
    result = store.code_peek("multiply two numbers")
    assert "multiply" in result or "Calculator" in result
    assert "return a * b" not in result


def test_provider_code_recall_and_peek(hermes_home: Path, sample_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fake_embedder import FakeEmbedder
    from turbomem import TurbomemMemoryProvider

    monkeypatch.setattr("turbomem.Embedder", FakeEmbedder)

    provider = TurbomemMemoryProvider()
    provider.initialize("test-session", hermes_home=str(hermes_home), agent_context="primary")
    provider.handle_tool_call("index_codebase", {"path": str(sample_repo)})

    recall = provider.handle_tool_call("code_recall", {"query": "add numbers"})
    assert "not implemented" not in recall.lower()
    assert "add" in recall or "math_utils" in recall

    peek = provider.handle_tool_call("code_peek", {"query": "add numbers"})
    assert "not implemented" not in peek.lower()
    assert "math_utils" in peek or "add" in peek
