"""Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence


def rrf_fuse(
    rankings: Sequence[Sequence[str]], k: int = 60, top: int | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(id) = sum over lists of 1 / (k + rank), rank starting at 1.

    Ties are broken by the best (lowest) rank in any list, then by id, so the result is
    deterministic.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for rank, rid in enumerate(ranking, start=1):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
            best_rank[rid] = min(best_rank.get(rid, rank), rank)
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))
    return fused if top is None else fused[:top]
