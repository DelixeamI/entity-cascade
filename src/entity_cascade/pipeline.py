"""Cascade stages 1-2 (retrieval + reranking) over a set of rows, with timings."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient

from entity_cascade.calibration import Calibration, calibrate
from entity_cascade.config import AppConfig
from entity_cascade.data.split import Splits
from entity_cascade.rerank.cross_encoder import Reranker
from entity_cascade.rerank.ranker import RawRow, score_rows
from entity_cascade.retrieval.candidates import CandidateGenerator, RowCandidates
from entity_cascade.retrieval.dense import Embedder
from entity_cascade.schemas import Record


@dataclass(frozen=True)
class RankingInputs:
    """Retrieval and cross-encoder output for rows; feature weights are applied later."""

    cands: dict[str, RowCandidates]
    raws: dict[str, RawRow]
    retrieval_ms_per_row: float
    rerank_ms_per_row: float


def fit_on_val(splits: Splits, val_raws: Mapping[str, RawRow], cfg: AppConfig) -> Calibration:
    """Fit feature weights and zone thresholds. Reads `splits.val` and nothing else."""
    return calibrate(splits.val, val_raws, cfg)


def build_generator(
    cfg: AppConfig, catalog: Sequence[Record], embedder: Embedder, client: QdrantClient
) -> tuple[CandidateGenerator, float]:
    """Build the BM25 and dense indexes. Returns the generator and index build seconds."""
    t0 = time.perf_counter()
    gen = CandidateGenerator(catalog, cfg, embedder, client)
    return gen, time.perf_counter() - t0


def benchmark_latency(
    cfg: AppConfig,
    gen: CandidateGenerator,
    catalog: Sequence[Record],
    rows: Sequence[Record],
    reranker: Reranker,
) -> tuple[float, float]:
    """(retrieval ms/row, rerank ms/row) on the given rows. Pass an uncached reranker."""
    inputs = compute_ranking_inputs(cfg, gen, catalog, rows, reranker)
    return inputs.retrieval_ms_per_row, inputs.rerank_ms_per_row


def compute_ranking_inputs(
    cfg: AppConfig,
    gen: CandidateGenerator,
    catalog: Sequence[Record],
    rows: Sequence[Record],
    reranker: Reranker,
) -> RankingInputs:
    catalog_by_id = {r.id: r for r in catalog}
    t0 = time.perf_counter()
    cands = gen.retrieve(rows)
    t1 = time.perf_counter()
    raws = score_rows(rows, cands, catalog_by_id, reranker, cfg)
    t2 = time.perf_counter()
    n = max(1, len(rows))
    return RankingInputs(
        cands=cands,
        raws=raws,
        retrieval_ms_per_row=1000 * (t1 - t0) / n,
        rerank_ms_per_row=1000 * (t2 - t1) / n,
    )
