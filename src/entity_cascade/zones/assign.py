"""Zone assignment from a ranking and the calibrated thresholds."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from entity_cascade.rerank.ranker import RowRanking
from entity_cascade.schemas import Zone

# Stand-in for "unreachable" that survives JSON (inf does not).
NEVER = 1e9


class Thresholds(BaseModel):
    """Cut-offs on the final score (cross-encoder logit + feature adjustment) and the margin.

    auto_match:   score >= high and margin >= margin_min
    llm_arbiter:  not auto, score >= low
    no_match:     not auto, score < floor
    human_review: the rest (floor <= score < low)
    """

    model_config = ConfigDict(frozen=True)

    high: float = NEVER
    margin_min: float = 0.0
    low: float = NEVER
    floor: float = -NEVER
    auto_feasible: bool = False  # False: no (high, margin) reached the target precision on val
    llm_feasible: bool = False
    no_match_feasible: bool = False


def assign_zone(ranking: RowRanking, t: Thresholds) -> Zone:
    if not ranking.candidates:
        return Zone.NO_MATCH
    s, m = ranking.score, ranking.margin
    if s >= t.high and m >= t.margin_min:
        return Zone.AUTO_MATCH
    if s >= t.low:
        return Zone.LLM_ARBITER
    if s < t.floor:
        return Zone.NO_MATCH
    return Zone.HUMAN_REVIEW
