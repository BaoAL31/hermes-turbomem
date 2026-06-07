from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, trust_remote_code=True)

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_model()
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def preload(self) -> None:
        self._ensure_model()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def is_cached(self) -> bool:
        return is_model_cached(self._model_name)


def is_model_cached(model_name: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=model_name, local_files_only=True)
        return True
    except Exception:
        return False
