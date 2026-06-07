from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

from hermes_turbomem.config import TurbomemConfig
from hermes_turbomem.store import MemoryStore

from fake_embedder import FakeEmbedder


def _install_agent_stub() -> None:
    if "agent.memory_provider" in sys.modules:
        return
    agent_mod = types.ModuleType("agent")
    memory_provider_mod = types.ModuleType("agent.memory_provider")

    class MemoryProvider:
        pass

    memory_provider_mod.MemoryProvider = MemoryProvider
    agent_mod.memory_provider = memory_provider_mod
    sys.modules["agent"] = agent_mod
    sys.modules["agent.memory_provider"] = memory_provider_mod


_install_agent_stub()


@pytest.fixture
def hermes_home(tmp_path: Path) -> Path:
    home = tmp_path / "hermes_home"
    home.mkdir()
    return home


@pytest.fixture
def turbomem_config(hermes_home: Path) -> TurbomemConfig:
    data_dir = hermes_home / "turbomem"
    return TurbomemConfig(data_dir=data_dir)


@pytest.fixture
def store(turbomem_config: TurbomemConfig) -> MemoryStore:
    return MemoryStore(turbomem_config, FakeEmbedder())


@pytest.fixture
def sample_repo() -> Path:
    repo = FIXTURES / "sample_repo"
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
