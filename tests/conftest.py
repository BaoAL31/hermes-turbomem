import shutil
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from hermes_turbomem.config import TurbomemConfig
from hermes_turbomem.store import MemoryStore


class FakeEmbedder:
    dimension: int = 64

    def encode(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        vecs = rng.normal(size=(len(texts), self.dimension)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms


@pytest.fixture()
def tmp_data_dir() -> Generator[Path, None, None]:
    tmp = Path(tempfile.mkdtemp(prefix="turbomem_test_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture()
def config(tmp_data_dir: Path) -> TurbomemConfig:
    return TurbomemConfig(
        data_dir=tmp_data_dir,
        auto_index_on_first_use=False,
        bit_width=4,
        default_recall_limit=8,
    )


@pytest.fixture()
def store(config: TurbomemConfig) -> MemoryStore:
    return MemoryStore(config, FakeEmbedder())


@pytest.fixture()
def sample_repo() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_repo"
