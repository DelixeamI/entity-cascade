"""Lexical retrieval with BM25 over normalized tokens."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from entity_cascade.data.normalize import tokenize


class BM25Index:
    def __init__(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts differ in length")
        self._ids = list(ids)
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._ids[i], float(scores[i])) for i in order]
