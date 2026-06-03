from __future__ import annotations

from pathlib import Path

import pytest

from hermes_turbomem.code_index import (
    CodeChunk,
    _is_denied,
    extract_chunks,
    file_content_hash,
    iter_indexable_files,
)
from hermes_turbomem.project_id import normalize_remote, resolve_project
from hermes_turbomem.config import TurbomemConfig, load_config


class TestProjectIdentity:
    def test_normalize_remote_https(self) -> None:
        assert normalize_remote("https://github.com/user/repo.git") == "github.com/user/repo"

    def test_normalize_remote_git(self) -> None:
        # SSH-style git@ URLs preserve colon between host and path
        result = normalize_remote("git@github.com:user/repo.git")
        assert result == "github.com:user/repo"
        assert "/" in result or ":" in result

    def test_normalize_remote_no_git_suffix(self) -> None:
        assert normalize_remote("https://github.com/user/repo") == "github.com/user/repo"

    def test_local_fallback(self, tmp_path: Path) -> None:
        info = resolve_project(tmp_path)
        assert info.project_id.startswith("local:")
        assert info.root == tmp_path.resolve()
        assert info.git_remote is None

    def test_git_repo(self, sample_repo: Path) -> None:
        info = resolve_project(sample_repo)
        assert info.project_id.startswith("local:") or info.project_id.startswith("git:")
        assert info.root == sample_repo.resolve()

    def test_resolve_from_child(self, sample_repo: Path) -> None:
        child = sample_repo / "utils" / "__init__.py"
        info = resolve_project(child)
        assert info.root == sample_repo.resolve()


class TestIndexScope:
    def test_deny_binary_extension(self) -> None:
        assert _is_denied("foo.bin")

    def test_deny_env_file(self) -> None:
        assert _is_denied(".env")
        assert _is_denied(".env.local")

    def test_deny_node_modules(self) -> None:
        assert _is_denied("node_modules")
        assert _is_denied("src/node_modules/foo.js")

    def test_deny_pycache(self) -> None:
        assert _is_denied("__pycache__/foo.pyc")

    def test_deny_build_dir(self) -> None:
        assert _is_denied("dist/bundle.js")

    def test_allow_normal_py_file(self) -> None:
        assert not _is_denied("math_utils.py")

    def test_allow_normal_js_file(self) -> None:
        assert not _is_denied("src/app.js")

    def test_gitignore_respected(self, sample_repo: Path) -> None:
        files = iter_indexable_files(sample_repo)
        rels = {f.relative_to(sample_repo).as_posix() for f in files}
        assert "ignored_file.py" not in rels
        assert "math_utils.py" in rels
        assert "string_utils.py" in rels
        assert "utils/__init__.py" in rels

    def test_denied_binary_excluded(self, sample_repo: Path) -> None:
        files = iter_indexable_files(sample_repo)
        rels = {str(f.relative_to(sample_repo)) for f in files}
        assert "denied_binary.bin" not in rels

    def test_sorted_output(self, sample_repo: Path) -> None:
        files = iter_indexable_files(sample_repo)
        rels = [str(f.relative_to(sample_repo)) for f in files]
        assert rels == sorted(rels)

    def test_only_code_extensions(self, sample_repo: Path) -> None:
        files = iter_indexable_files(sample_repo)
        for f in files:
            assert f.suffix in {".py", ".js", ".ts", ".go", ".rs", ".java", ".rb", ".cs", ".cpp", ".c", ".h", ".hpp", ".jsx", ".tsx"}


class TestChunking:
    def test_content_hash_deterministic(self) -> None:
        assert file_content_hash("hello") == file_content_hash("hello")
        assert file_content_hash("hello") != file_content_hash("world")

    def test_extract_chunks_python_file(self, sample_repo: Path) -> None:
        file_path = sample_repo / "math_utils.py"
        chunks = extract_chunks(file_path, sample_repo)
        chunk_symbols = [c.symbol for c in chunks]
        assert "add" in chunk_symbols
        assert "subtract" in chunk_symbols
        assert "Calculator" in chunk_symbols

    def test_chunk_has_location(self, sample_repo: Path) -> None:
        file_path = sample_repo / "math_utils.py"
        chunks = extract_chunks(file_path, sample_repo)
        for c in chunks:
            assert c.start_line >= 1
            assert c.end_line >= c.start_line
            assert c.path == "math_utils.py"

    def test_chunk_has_text(self, sample_repo: Path) -> None:
        file_path = sample_repo / "math_utils.py"
        chunks = extract_chunks(file_path, sample_repo)
        for c in chunks:
            assert len(c.text) > 0
            assert c.content_hash


class TestConfig:
    def test_default_values(self) -> None:
        cfg = TurbomemConfig()
        assert cfg.bit_width == 4
        assert cfg.default_recall_limit == 8
        assert cfg.embedding_model == "nomic-ai/nomic-embed-text-v1"
        assert not cfg.auto_index_on_first_use

    def test_catalog_path_under_data_dir(self) -> None:
        cfg = TurbomemConfig(data_dir=Path("/tmp/turbomem"))
        assert cfg.catalog_path == Path("/tmp/turbomem/catalog.db")

    def test_custom_values(self) -> None:
        cfg = TurbomemConfig(bit_width=8, default_recall_limit=16)
        assert cfg.bit_width == 8
        assert cfg.default_recall_limit == 16
