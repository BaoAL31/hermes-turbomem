from __future__ import annotations

import hashlib

import numpy as np


class FakeEmbedder:
    """Deterministic embedder for tests — no model download."""

    dimension = 16

    def __init__(self, model_name: str = "test/fake-embed") -> None:
        self._model_name = model_name
        self._loaded = False

    @property
    def model_name(self) -> str:
        return self._model_name

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        tiled = np.tile(raw, int(np.ceil(self.dimension / raw.size)))[: self.dimension]
        norm = np.linalg.norm(tiled)
        if norm == 0:
            return tiled
        return tiled / norm

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts], axis=0)

    def preload(self) -> None:
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def is_cached(self) -> bool:
        return True
