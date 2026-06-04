from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from hermes_turbomem.diagnostics import get_logger, get_metrics


class Embedder:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name, trust_remote_code=True)
        self._log = get_logger()
        self._metrics = get_metrics()

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        self._metrics.increment("embed_call")
        try:
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(vectors, dtype=np.float32)
        except Exception as exc:
            self._metrics.increment("embed_error")
            self._log.log("embed", "ERROR", f"Embedding failed: {exc}")
            raise
