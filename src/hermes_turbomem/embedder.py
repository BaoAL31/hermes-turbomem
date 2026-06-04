from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name, trust_remote_code=True)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def preload(self) -> None:
        _ = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
