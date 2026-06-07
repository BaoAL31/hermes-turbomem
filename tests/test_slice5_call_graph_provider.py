from __future__ import annotations

from pathlib import Path

import pytest


def test_provider_code_call_graph(hermes_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from fake_embedder import FakeEmbedder
    from turbomem import TurbomemMemoryProvider

    monkeypatch.setattr("turbomem.Embedder", FakeEmbedder)

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text(
        "def helper() -> int:\n    return 42\n\n"
        "def run() -> int:\n    return helper()\n"
    )

    provider = TurbomemMemoryProvider()
    provider.initialize("test-session", hermes_home=str(hermes_home), agent_context="primary")
    provider.handle_tool_call("index_codebase", {"path": str(repo)})

    callees = provider.handle_tool_call("code_call_graph", {"name": "run", "direction": "callees"})
    assert "not implemented" not in callees.lower()
    assert "helper" in callees

    callers = provider.handle_tool_call("code_call_graph", {"name": "helper", "direction": "callers"})
    assert "run" in callers
