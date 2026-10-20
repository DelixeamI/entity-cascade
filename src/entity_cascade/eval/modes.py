"""Compare retrieval/ranking modes on equal terms: each mode is trusted by its own score+margin."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from entity_cascade.config import ThresholdsConfig
from entity_cascade.data.split import TestSplit, ValSplit
from entity_cascade.eval.types import NanFloat
from entity_cascade.rerank.features import FeatureWeights
from entity_cascade.rerank.ranker import RawRow, rank_row
from entity_cascade.retrieval.candidates import RowCandidates
from entity_cascade.zones.thresholds import ScoredTop1, fit_auto_only

MODES = (
    "bm25",
    "dense",
    "hybrid (RRF)",
    "hybrid + reranker",
    "hybrid + reranker + features",
)


class ModeRow(BaseModel):
    mode: str
    top1_accuracy_test: NanFloat  # rows with a pair
    feasible: bool  # a val threshold reaching the target precision exists
    val_auto_share: NanFloat
    val_auto_precision: NanFloat
    test_auto_share: NanFloat
    test_auto_precision: NanFloat


def _top_two(ids: Sequence[str], scores: Sequence[float]) -> ScoredTop1:
    if not ids or not scores:
        return ScoredTop1(None, float("-inf"), 0.0)
    margin = scores[0] - scores[1] if len(scores) > 1 else scores[0]
    return ScoredTop1(ids[0], float(scores[0]), float(margin))


def scored_for_mode(
    mode: str,
    row_ids: Sequence[str],
    cands: Mapping[str, RowCandidates],
    raws: Mapping[str, RawRow],
    weights: FeatureWeights,
) -> dict[str, ScoredTop1]:
    out: dict[str, ScoredTop1] = {}
    for rid in row_ids:
        c = cands[rid]
        if mode == "bm25":
            out[rid] = _top_two(c.bm25, c.bm25_scores)
        elif mode == "dense":
            out[rid] = _top_two(c.dense, c.dense_scores)
        elif mode == "hybrid (RRF)":
            out[rid] = _top_two(c.hybrid, c.hybrid_scores)
        else:
            w = FeatureWeights() if mode == "hybrid + reranker" else weights
            rk = rank_row(raws[rid], w)
            out[rid] = ScoredTop1(rk.top1_id, rk.score, rk.margin)
    return out


def _zone_stats(
    scored: Mapping[str, ScoredTop1],
    truth: Mapping[str, frozenset[str]],
    high: float,
    margin_min: float,
) -> tuple[float, float]:
    """(share of all rows auto-resolved, precision of those)."""
    sel = [
        rid
        for rid, s in scored.items()
        if s.top1_id is not None and s.score >= high and s.margin >= margin_min
    ]
    n = len(scored)
    share = len(sel) / n if n else float("nan")
    prec = sum(scored[r].top1_id in truth[r] for r in sel) / len(sel) if sel else float("nan")
    return share, prec


def mode_row(
    mode: str,
    val: ValSplit,
    test: TestSplit,
    scored_val: Mapping[str, ScoredTop1],
    scored_test: Mapping[str, ScoredTop1],
    cfg: ThresholdsConfig,
) -> ModeRow:
    high, mm, ok = fit_auto_only(val, scored_val, cfg)  # val only
    pair_rows = [r.id for r in test.rows if test.truth[r.id]]
    acc = (
        sum(scored_test[r].top1_id in test.truth[r] for r in pair_rows) / len(pair_rows)
        if pair_rows
        else float("nan")
    )
    nan = float("nan")
    v_share, v_prec = _zone_stats(scored_val, val.truth, high, mm) if ok else (0.0, nan)
    t_share, t_prec = _zone_stats(scored_test, test.truth, high, mm) if ok else (0.0, nan)
    return ModeRow(
        mode=mode,
        top1_accuracy_test=acc,
        feasible=ok,
        val_auto_share=v_share,
        val_auto_precision=v_prec,
        test_auto_share=t_share,
        test_auto_precision=t_prec,
    )


@dataclass(frozen=True)
class CurvePoint:
    coverage: NanFloat  # share of all rows that clear the threshold
    precision: NanFloat  # of those, share whose top-1 is right


def precision_coverage_curve(
    scored: Mapping[str, ScoredTop1], truth: Mapping[str, frozenset[str]]
) -> list[CurvePoint]:
    """Sweep a score threshold from strict to loose; margin gate off."""
    items = sorted(
        ((s.score, s.top1_id in truth[rid]) for rid, s in scored.items() if s.top1_id is not None),
        key=lambda x: -x[0],
    )
    n_all = len(scored)
    scores = np.array([s for s, _ in items])
    correct = np.cumsum([c for _, c in items])
    points: list[CurvePoint] = []
    for i in range(len(items)):
        last_of_tie = i == len(items) - 1 or scores[i + 1] != scores[i]
        if last_of_tie:
            points.append(CurvePoint((i + 1) / n_all, float(correct[i]) / (i + 1)))
    return points
