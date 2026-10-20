"""Cross-encoder scoring of retrieved candidates, then feature-adjusted ranking with a margin."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from entity_cascade.config import AppConfig
from entity_cascade.rerank.cross_encoder import Reranker
from entity_cascade.rerank.features import FeatureWeights, MatchFeatures, compute_features
from entity_cascade.retrieval.candidates import RowCandidates
from entity_cascade.retrieval.text import record_text
from entity_cascade.schemas import Record


@dataclass(frozen=True)
class RawCandidate:
    """Everything that does not depend on feature weights."""

    catalog_id: str
    hybrid_rank: int  # 1-based position in the RRF list
    ce_score: float
    features: MatchFeatures


@dataclass(frozen=True)
class RawRow:
    row_id: str
    items: tuple[RawCandidate, ...]  # in RRF order


@dataclass(frozen=True)
class ScoredCandidate:
    catalog_id: str
    hybrid_rank: int
    ce_score: float
    features: MatchFeatures
    final_score: float  # ce_score + feature adjustment


@dataclass(frozen=True)
class RowRanking:
    row_id: str
    candidates: tuple[ScoredCandidate, ...]  # best first

    @property
    def top1_id(self) -> str | None:
        return self.candidates[0].catalog_id if self.candidates else None

    @property
    def score(self) -> float:
        return self.candidates[0].final_score if self.candidates else float("-inf")

    @property
    def margin(self) -> float:
        """Top-1 minus top-2 final score. With a single candidate there is no competitor and
        the margin is the score itself (as if the runner-up scored 0). Empty list: 0."""
        if not self.candidates:
            return 0.0
        if len(self.candidates) == 1:
            return self.candidates[0].final_score
        return self.candidates[0].final_score - self.candidates[1].final_score


def score_rows(
    rows: Sequence[Record],
    cands: Mapping[str, RowCandidates],
    catalog_by_id: Mapping[str, Record],
    reranker: Reranker,
    cfg: AppConfig,
) -> dict[str, RawRow]:
    """Score every (row, candidate) pair in one batch run."""
    pairs: list[tuple[str, str]] = []
    index: list[tuple[str, int, str]] = []
    feats: dict[tuple[str, str], MatchFeatures] = {}
    for row in rows:
        query = record_text(row, cfg.rerank.query_fields)
        for rank, cid in enumerate(cands[row.id].hybrid, start=1):
            item = catalog_by_id[cid]
            pairs.append((query, record_text(item, cfg.dataset.right_text)))
            index.append((row.id, rank, cid))
            feats[(row.id, cid)] = compute_features(row, item)
    scores = reranker.score(pairs)
    if len(scores) != len(pairs):
        raise RuntimeError("reranker returned a different number of scores than pairs")
    grouped: dict[str, list[RawCandidate]] = {row.id: [] for row in rows}
    for (rid, rank, cid), s in zip(index, scores, strict=True):
        grouped[rid].append(RawCandidate(cid, rank, s, feats[(rid, cid)]))
    return {rid: RawRow(rid, tuple(items)) for rid, items in grouped.items()}


def rank_row(raw: RawRow, weights: FeatureWeights) -> RowRanking:
    scored = [
        ScoredCandidate(
            catalog_id=c.catalog_id,
            hybrid_rank=c.hybrid_rank,
            ce_score=c.ce_score,
            features=c.features,
            final_score=c.ce_score + weights.adjust(c.features),
        )
        for c in raw.items
    ]
    scored.sort(key=lambda c: (-c.final_score, c.hybrid_rank))
    return RowRanking(raw.row_id, tuple(scored))
