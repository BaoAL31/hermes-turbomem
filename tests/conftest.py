from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_repo() -> Path:
    return FIXTURES / "sample_repo"


@pytest.fixture
def utils_project() -> Path:
    return FIXTURES / "utils_project"


@pytest.fixture
def tmp_catalog(tmp_path: Path) -> Path:
    return tmp_path / "catalog"
