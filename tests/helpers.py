"""Test doubles shared by several test modules."""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from entity_cascade.data.normalize import tokenize


class HashingEmbedder:
    """Deterministic bag-of-tokens embedder: no model download, enough for plumbing tests."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                out[i, zlib.crc32(tok.encode()) % self._dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return np.asarray(out / norms, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._embed(texts)

    def embed_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._embed(texts)
