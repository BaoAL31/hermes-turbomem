from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def test_store_reads_config_yaml(hermes_home: Path) -> None:
    from hermes_turbomem.config import load_config

    turbomem_dir = hermes_home / "turbomem"
    turbomem_dir.mkdir(parents=True)
    (turbomem_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "auto_index_on_first_use": False,
                "embedding_model": "test/custom-model",
                "bit_width": 8,
                "default_recall_limit": 12,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg = load_config(turbomem_dir)
    assert cfg.auto_index_on_first_use is False
    assert cfg.embedding_model == "test/custom-model"
    assert cfg.bit_width == 8
    assert cfg.default_recall_limit == 12


def test_provider_reads_provider_yaml_retain_knobs(
    hermes_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    from fake_embedder import FakeEmbedder
    from turbomem import TurbomemMemoryProvider

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
    monkeypatch.setattr("turbomem.Embedder", FakeEmbedder)

    turbomem_dir = hermes_home / "turbomem"
    turbomem_dir.mkdir(parents=True)
    (turbomem_dir / "provider.yaml").write_text(
        yaml.safe_dump(
            {
                "retain_every_n_turns": 3,
                "retain_max_chars_per_side": 500,
                "recall_max_chars": 2000,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    provider = TurbomemMemoryProvider()
    provider.initialize("s1", hermes_home=str(hermes_home), agent_context="primary")
    assert provider._retain_every_n_turns == 3
    assert provider._retain_max_chars_per_side == 500
    assert provider._recall_max_chars == 2000

    provider.sync_turn("a", "b")
    provider.sync_turn("c", "d")
    provider.sync_turn("e", "f")
    store = provider._store
    assert store is not None
    convos = store.list_experiences(category="conversation")
    assert "User: e" in convos
    assert "User: a" not in convos
