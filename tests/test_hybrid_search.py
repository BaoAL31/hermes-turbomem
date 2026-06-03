from __future__ import annotations

import numpy as np
import pytest
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from hermes_turbomem.bm25 import BM25Index, rrf_fuse


FakeRowData = dict[str, object]


class FakeRow:
    """Simulates sqlite3.Row for format tests."""

    def __init__(self, data: FakeRowData) -> None:
        self._data = data

    def __getitem__(self, key: str) -> object:
        return self._data[key]


# ---------------------------------------------------------------------------
# BM25Index unit tests
# ---------------------------------------------------------------------------


class TestBM25Index:
    def test_tokenize(self) -> None:
        assert BM25Index.tokenize("hello world") == ["hello", "world"]
        assert BM25Index.tokenize("camelCase snake_case") == ["camelcase", "snake_case"]
        assert BM25Index.tokenize("add_numbers(a, b)") == ["add_numbers", "a", "b"]
        assert BM25Index.tokenize("") == []

    def test_empty_index_returns_empty(self) -> None:
        idx = BM25Index()
        assert idx.search("hello") == []

    def test_exact_match_ranks_highest(self) -> None:
        idx = BM25Index()
        idx.add_document(1, "def add(a, b): return a + b", "add")
        idx.add_document(2, "def subtract(a, b): return a - b", "subtract")
        idx.add_document(3, "class Calculator:", "Calculator")

        results = idx.search("add", top_k=5)
        assert len(results) >= 1
        top_id, _ = results[0]
        assert top_id == 1

    def test_symbol_name_match_ranks_high(self) -> None:
        idx = BM25Index()
        idx.add_document(1, "def process_data(): pass", "process_data")
        idx.add_document(2, "def transform(items): pass", "transform")
        idx.add_document(3, "def validate_input(): pass", "validate_input")

        results = idx.search("transform", top_k=5)
        assert len(results) >= 1
        top_id, _ = results[0]
        assert top_id == 2

    def test_query_with_multiple_terms(self) -> None:
        idx = BM25Index()
        idx.add_document(1, "Load and parse configuration from YAML file", "load_config")
        idx.add_document(2, "Save user preferences to disk", "save_prefs")
        idx.add_document(3, "Parse command line arguments", "parse_args")

        results = idx.search("parse config YAML", top_k=5)
        assert len(results) >= 2

    def test_top_k_limits_results(self) -> None:
        idx = BM25Index()
        for i in range(10):
            idx.add_document(i, f"function_{i} does something with data", f"func_{i}")

        results = idx.search("data something", top_k=3)
        assert len(results) <= 3

    def test_no_match_returns_empty(self) -> None:
        idx = BM25Index()
        idx.add_document(1, "def python_function(): pass", "python_function")
        idx.add_document(2, "def rust_function(): pass", "rust_function")

        results = idx.search("zzz_nonexistent", top_k=5)
        assert results == []

    def test_extend_adds_all(self) -> None:
        idx = BM25Index()
        idx.extend([
            (1, "def alpha(): pass", "alpha"),
            (2, "def beta(): pass", "beta"),
        ])
        assert idx.num_docs == 2
        results = idx.search("alpha", top_k=5)
        assert results[0][0] == 1


# ---------------------------------------------------------------------------
# RRF fusion unit tests
# ---------------------------------------------------------------------------


class TestRRFFusion:
    def test_both_lists_empty(self) -> None:
        assert rrf_fuse([], []) == []

    def test_only_semantic(self) -> None:
        result = rrf_fuse([(1, 0.9), (2, 0.8)], [])
        assert len(result) == 2
        assert result[0][0] == 1
        assert result[1][0] == 2

    def test_only_bm25(self) -> None:
        result = rrf_fuse([], [(2, 5.0), (1, 3.0)])
        assert len(result) == 2
        assert result[0][0] == 2
        assert result[1][0] == 1

    def test_doc_in_both_gets_boost(self) -> None:
        result = rrf_fuse(
            [(1, 0.9), (2, 0.8), (3, 0.7)],
            [(2, 10.0), (3, 5.0), (1, 2.0)],
        )
        ids = [eid for eid, _ in result]
        assert 2 in ids
        assert 1 in ids
        assert 3 in ids

    def test_rrf_prioritizes_shared_high_rank(self) -> None:
        semantic = [(1, 0.9), (2, 0.8), (3, 0.7)]
        bm25 = [(2, 10.0), (3, 5.0), (4, 1.0)]
        result = rrf_fuse(semantic, bm25)
        ids = [eid for eid, _ in result]
        assert ids[0] == 2

    def test_custom_k(self) -> None:
        semantic = [(1, 0.9)]
        bm25 = [(2, 10.0)]
        result_default = rrf_fuse(semantic, bm25)
        result_custom = rrf_fuse(semantic, bm25, k=1)
        assert result_custom[0][1] != result_default[0][1]


# ---------------------------------------------------------------------------
# Formatting tests
# ---------------------------------------------------------------------------


def _make_fake_config(**overrides: object) -> object:
    from hermes_turbomem.config import TurbomemConfig
    return TurbomemConfig(**overrides)


class _FakeEmbedder:
    dimension = 64

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 64), dtype=np.float32)


def _make_fake_embedder() -> _FakeEmbedder:
    return _FakeEmbedder()


class TestFormatHit:
    def test_code_low_confidence(self) -> None:
        from hermes_turbomem.store import MemoryStore

        store = MemoryStore.__new__(MemoryStore)
        store.config = _make_fake_config()
        store.embedder = _make_fake_embedder()

        row = FakeRow({
            "entry_type": "code",
            "project_id": "test-proj",
            "symbol": "my_function",
            "path": "src/mod.py",
            "start_line": 10,
            "end_line": 25,
            "text": "def my_function(): pass",
        })

        result = store._format_hit(row, 0.15, is_low=True)
        assert "[LOW-CONFIDENCE]" in result
        assert "my_function @ src/mod.py:10-25" in result
        assert "def my_function(): pass" in result

    def test_code_normal_confidence(self) -> None:
        from hermes_turbomem.store import MemoryStore

        store = MemoryStore.__new__(MemoryStore)
        store.config = _make_fake_config()
        store.embedder = _make_fake_embedder()

        row = FakeRow({
            "entry_type": "code",
            "project_id": "test-proj",
            "symbol": "my_function",
            "path": "src/mod.py",
            "start_line": 10,
            "end_line": 25,
            "text": "def my_function(): pass",
        })

        result = store._format_hit(row, 0.45, is_low=False)
        assert "[LOW-CONFIDENCE]" not in result
        assert "my_function @ src/mod.py:10-25" in result

    def test_experience_low_confidence(self) -> None:
        from hermes_turbomem.store import MemoryStore

        store = MemoryStore.__new__(MemoryStore)
        store.config = _make_fake_config()
        store.embedder = _make_fake_embedder()

        row = FakeRow({
            "entry_type": "experience",
            "category": "preference",
            "project_id": None,
            "text": "Use yarn not npm",
            "symbol": None,
            "path": None,
            "start_line": None,
            "end_line": None,
        })

        result = store._format_hit(row, 0.25, is_low=True)
        assert "[LOW-CONFIDENCE]" in result
        assert "Use yarn not npm" in result


class TestFormatPeekHit:
    def test_code_omits_source_body(self) -> None:
        from hermes_turbomem.store import MemoryStore

        store = MemoryStore.__new__(MemoryStore)
        store.config = _make_fake_config()
        store.embedder = _make_fake_embedder()

        row = FakeRow({
            "entry_type": "code",
            "project_id": "test-proj",
            "symbol": "my_function",
            "path": "src/mod.py",
            "start_line": 10,
            "end_line": 25,
            "text": "def my_function(): pass",
        })

        result = store._format_peek_hit(row, 0.45, is_low=False)
        assert "[LOW-CONFIDENCE]" not in result
        assert "my_function @ src/mod.py:10-25" in result
        assert "def my_function(): pass" not in result

    def test_code_low_confidence(self) -> None:
        from hermes_turbomem.store import MemoryStore

        store = MemoryStore.__new__(MemoryStore)
        store.config = _make_fake_config()
        store.embedder = _make_fake_embedder()

        row = FakeRow({
            "entry_type": "code",
            "project_id": "test-proj",
            "symbol": "weak_match",
            "path": "src/other.py",
            "start_line": 5,
            "end_line": 8,
            "text": "some code here",
        })

        result = store._format_peek_hit(row, 0.15, is_low=True)
        assert "[LOW-CONFIDENCE]" in result
        assert "weak_match @ src/other.py:5-8" in result

    def test_experience_peek(self) -> None:
        from hermes_turbomem.store import MemoryStore

        store = MemoryStore.__new__(MemoryStore)
        store.config = _make_fake_config()
        store.embedder = _make_fake_embedder()

        row = FakeRow({
            "entry_type": "experience",
            "category": "preference",
            "project_id": None,
            "text": "Use yarn not npm",
            "symbol": None,
            "path": None,
            "start_line": None,
            "end_line": None,
        })

        result = store._format_peek_hit(row, 0.25, is_low=True)
        assert "[LOW-CONFIDENCE]" in result
        assert "(peek)" in result
