"""Dense retrieval: sentence-transformers embeddings stored in Qdrant."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    def embed_queries(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class SentenceTransformerEmbedder:
    """bge-style embedder: L2-normalized vectors, optional instruction prefix for queries."""

    def __init__(self, model_name: str, query_instruction: str = "", batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self._instruction = query_instruction
        self._batch_size = batch_size

    @property
    def dim(self) -> int:
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError("embedding model reports no dimension")
        return int(dim)

    def _encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vecs = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._encode(texts)

    def embed_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._encode([self._instruction + t for t in texts])


def make_client(url: str) -> QdrantClient:
    """ ":memory:" gives an in-process Qdrant (tests, CI, eval); otherwise an HTTP URL."""
    return QdrantClient(":memory:") if url == ":memory:" else QdrantClient(url=url)


class DenseIndex:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder

    def build(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts differ in length")
        vectors = self._embedder.embed_documents(texts)
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            self._collection,
            vectors_config=VectorParams(size=self._embedder.dim, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=i, vector=vectors[i].tolist(), payload={"rid": rid})
            for i, rid in enumerate(ids)
        ]
        for start in range(0, len(points), 256):
            self._client.upsert(self._collection, points=points[start : start + 256])

    def search_many(self, queries: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        if not queries:
            return []
        vectors = self._embedder.embed_queries(queries)
        out: list[list[tuple[str, float]]] = []
        for vec in vectors:
            res = self._client.query_points(self._collection, query=vec.tolist(), limit=k)
            out.append([(str((p.payload or {})["rid"]), float(p.score)) for p in res.points])
        return out
