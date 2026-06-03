from __future__ import annotations

import math
import re
from collections import Counter


class BM25Index:
    """BM25-Okapi ranker over a static text collection.

    Call add_document() for each document, then search() to query.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[tuple[int, str, str | None]] = []
        self._built = False

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z_]\w*", text.lower())

    def add_document(self, entry_id: int, text: str, symbol: str | None = None) -> None:
        self._docs.append((entry_id, text, symbol))
        self._built = False

    def extend(self, docs: list[tuple[int, str, str | None]]) -> None:
        self._docs.extend(docs)
        self._built = False

    def clear(self) -> None:
        self._docs.clear()
        self._built = False

    @property
    def num_docs(self) -> int:
        return len(self._docs)

    def _build(self) -> None:
        if self._built:
            return

        n = len(self._docs)
        self._doc_freqs: list[Counter[str]] = [Counter() for _ in range(n)]
        self._doc_lengths: list[int] = [0] * n
        term_doc_count: dict[str, int] = {}

        for i, (_, text, symbol) in enumerate(self._docs):
            tokens = self.tokenize(f"{symbol} {text}") if symbol else self.tokenize(text)
            freq = Counter(tokens)
            self._doc_freqs[i] = freq
            self._doc_lengths[i] = len(tokens)
            for term in freq:
                term_doc_count[term] = term_doc_count.get(term, 0) + 1

        self._avgdl = sum(self._doc_lengths) / n if n > 0 else 0.0
        self._idf: dict[str, float] = {}
        for term, df in term_doc_count.items():
            self._idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

        self._built = True

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if not self._docs:
            return []
        self._build()

        query_terms = self.tokenize(query)
        if not query_terms:
            return []

        scores: dict[int, float] = {}
        for term in query_terms:
            idf_val = self._idf.get(term, 0.0)
            if idf_val == 0.0:
                continue
            for i, freq in enumerate(self._doc_freqs):
                tf = freq.get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_lengths[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                scores[self._docs[i][0]] = scores.get(self._docs[i][0], 0.0) + idf_val * numerator / denominator

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def rrf_fuse(
    semantic: list[tuple[int, float]],
    bm25: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion of two ranked result lists.

    Each input is a list of (entry_id, score) pairs ordered by descending score.
    Returns merged list ordered by descending RRF score.
    """
    rrf_scores: dict[int, float] = {}
    for rank, (eid, _) in enumerate(semantic):
        rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (eid, _) in enumerate(bm25):
        rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
