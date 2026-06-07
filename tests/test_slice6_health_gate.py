from __future__ import annotations

from pathlib import Path

import pytest

from hermes_turbomem.config import TurbomemConfig
from hermes_turbomem.store import MemoryStore


def test_embed_config_mismatch_blocks_code_recall(store: MemoryStore, sample_repo: Path) -> None:
    store.index_codebase(str(sample_repo))
    store._conn.execute(
        "INSERT INTO index_metadata (key, value) VALUES ('bit_width', '99') "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
    )
    store._conn.commit()
    result = store.code_recall("add numbers")
    assert "mismatch" in result.lower() or "metadata" in result.lower()


def test_provider_index_health_check(hermes_home: Path, sample_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fake_embedder import FakeEmbedder
    from turbomem import TurbomemMemoryProvider

    monkeypatch.setattr("turbomem.Embedder", FakeEmbedder)

    provider = TurbomemMemoryProvider()
    provider.initialize("test-session", hermes_home=str(hermes_home), agent_context="primary")
    provider.handle_tool_call("index_codebase", {"path": str(sample_repo)})
    result = provider.handle_tool_call("index_health_check", {})
    assert "not implemented" not in result.lower()
    assert "health check" in result.lower()
