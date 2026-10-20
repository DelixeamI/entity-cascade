"""Evaluation metrics. Pure functions over rankings and ground truth."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

DEFAULT_KS = (1, 5, 20)


def top1_accuracy(
    top1: Mapping[str, str | None], truth: Mapping[str, frozenset[str]]
) -> tuple[float, int]:
    """Share of rows with a true pair whose top-1 is a true item. Returns (accuracy, n_rows)."""
    rows = [rid for rid, t in truth.items() if t]
    if not rows:
        return float("nan"), 0
    hits = sum(1 for rid in rows if top1.get(rid) in truth[rid])
    return hits / len(rows), len(rows)


def candidate_recall(
    rankings: Mapping[str, Sequence[str]],
    truth: Mapping[str, frozenset[str]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[int, float]:
    """Share of rows with a true pair whose top-k contains at least one true item.

    Rows without a true pair (new products) are excluded: no ranking can hit for them.
    """
    rows = [rid for rid, t in truth.items() if t]
    if not rows:
        return {k: float("nan") for k in ks}
    out: dict[int, float] = {}
    for k in ks:
        hits = sum(1 for rid in rows if truth[rid] & set(rankings.get(rid, ())[:k]))
        out[k] = hits / len(rows)
    return out
