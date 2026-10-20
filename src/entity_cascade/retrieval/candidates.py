"""Candidate generation: BM25, dense, and their RRF fusion over the catalog."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient

from entity_cascade.config import AppConfig
from entity_cascade.retrieval.bm25 import BM25Index
from entity_cascade.retrieval.dense import DenseIndex, Embedder
from entity_cascade.retrieval.rrf import rrf_fuse
from entity_cascade.retrieval.text import record_text
from entity_cascade.schemas import Record


@dataclass(frozen=True)
class RowCandidates:
    row_id: str
    bm25: tuple[str, ...]  # catalog ids, best first, up to retrieval.per_mode_k
    dense: tuple[str, ...]
    hybrid: tuple[str, ...]  # RRF-fused, up to retrieval.top_k
    hybrid_scores: tuple[float, ...]
    bm25_scores: tuple[float, ...] = ()  # aligned with bm25
    dense_scores: tuple[float, ...] = ()  # aligned with dense (cosine)


class CandidateGenerator:
    def __init__(
        self,
        catalog: Sequence[Record],
        cfg: AppConfig,
        embedder: Embedder,
        client: QdrantClient,
    ) -> None:
        self._cfg = cfg
        ids = [r.id for r in catalog]
        texts = [record_text(r, cfg.dataset.right_text) for r in catalog]
        self._bm25 = BM25Index(ids, texts)
        self._dense = DenseIndex(client, cfg.retrieval.collection, embedder)
        self._dense.build(ids, texts)

    def retrieve(self, rows: Sequence[Record]) -> dict[str, RowCandidates]:
        rc = self._cfg.retrieval
        queries = [record_text(r, self._cfg.dataset.left_text) for r in rows]
        dense_hits = self._dense.search_many(queries, rc.per_mode_k)
        out: dict[str, RowCandidates] = {}
        for row, query, dense in zip(rows, queries, dense_hits, strict=True):
            bm25_hits = self._bm25.search(query, rc.per_mode_k)
            bm25_ids = tuple(rid for rid, _ in bm25_hits)
            dense_ids = tuple(rid for rid, _ in dense)
            fused = rrf_fuse([bm25_ids, dense_ids], k=rc.rrf_k, top=rc.top_k)
            out[row.id] = RowCandidates(
                row_id=row.id,
                bm25=bm25_ids,
                dense=dense_ids,
                hybrid=tuple(rid for rid, _ in fused),
                hybrid_scores=tuple(s for _, s in fused),
                bm25_scores=tuple(s for _, s in bm25_hits),
                dense_scores=tuple(s for _, s in dense),
            )
        return out
