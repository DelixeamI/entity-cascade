"""Choose zone thresholds on val. Takes a ValSplit by type: test rows cannot be passed in."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from entity_cascade.config import ThresholdsConfig
from entity_cascade.data.split import ValSplit
from entity_cascade.rerank.ranker import RowRanking
from entity_cascade.zones.assign import NEVER, Thresholds

ARBITER_SHOWN = 3  # candidates the LLM sees; must match llm.top_candidates in the prompt
MARGIN_QUANTILES = tuple(np.linspace(0.1, 0.9, 9))


def wilson_lower(successes: int, n: int, z: float) -> float:
    """Lower end of the Wilson score interval. z = 0 gives the plain proportion."""
    if n == 0:
        return 0.0
    p = successes / n
    if z == 0.0:
        return p
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre - margin) / denom)


def fit_thresholds(
    val: ValSplit,
    rankings: Mapping[str, RowRanking],
    cfg: ThresholdsConfig,
    shown: int = ARBITER_SHOWN,
) -> Thresholds:
    rows = [r for r in val.rows if rankings[r.id].candidates]
    if not rows:
        return Thresholds()
    score = np.array([rankings[r.id].score for r in rows])
    margin = np.array([rankings[r.id].margin for r in rows])
    truth = [val.truth[r.id] for r in rows]
    correct = np.array([rankings[r.id].top1_id in t for r, t in zip(rows, truth, strict=True)])
    has_pair = np.array([bool(t) for t in truth])
    in_top = np.array(
        [
            bool(t & {c.catalog_id for c in rankings[r.id].candidates[:shown]})
            for r, t in zip(rows, truth, strict=True)
        ]
    )
    answerable = in_top | ~has_pair  # the arbiter can be right: item shown, or "none" is right

    auto_mask, high, margin_min, auto_ok = _fit_auto(score, margin, correct, cfg)
    rest = ~auto_mask
    floor, floor_ok = _fit_floor(score, has_pair, rest, cfg)
    low, low_ok = _fit_low(score, answerable, rest, floor, cfg)
    return Thresholds(
        high=high,
        margin_min=margin_min,
        low=low,
        floor=floor,
        auto_feasible=auto_ok,
        llm_feasible=low_ok,
        no_match_feasible=floor_ok,
    )


@dataclass(frozen=True)
class ScoredTop1:
    """One mode's answer for a row: its top-1 plus the score and margin it is trusted by."""

    top1_id: str | None
    score: float
    margin: float


def fit_auto_only(
    val: ValSplit, scored: Mapping[str, ScoredTop1], cfg: ThresholdsConfig
) -> tuple[float, float, bool]:
    """Auto zone only, for any retrieval mode. Returns (high, margin_min, feasible)."""
    rows = [r for r in val.rows if scored[r.id].top1_id is not None]
    if not rows:
        return NEVER, 0.0, False
    score = np.array([scored[r.id].score for r in rows])
    margin = np.array([scored[r.id].margin for r in rows])
    correct = np.array([scored[r.id].top1_id in val.truth[r.id] for r in rows])
    _, high, margin_min, ok = _fit_auto(score, margin, correct, cfg)
    return high, margin_min, ok


def _fit_auto(
    score: NDArray[np.float64],
    margin: NDArray[np.float64],
    correct: NDArray[np.bool_],
    cfg: ThresholdsConfig,
) -> tuple[NDArray[np.bool_], float, float, bool]:
    """Largest auto_match zone whose precision (lower bound) reaches the target."""
    margins = sorted({0.0, *(float(q) for q in np.quantile(margin, MARGIN_QUANTILES))})
    best: tuple[int, float, float, float] | None = None  # (n, precision, high, margin_min)
    for h in sorted({float(s) for s in score}, reverse=True):
        for mm in margins:
            mask = (score >= h) & (margin >= mm)
            n = int(mask.sum())
            if n < cfg.min_auto_support:
                continue
            k = int(correct[mask].sum())
            if wilson_lower(k, n, cfg.wilson_z) < cfg.target_precision:
                continue
            cand = (n, k / n, h, mm)
            if best is None or cand > best:
                best = cand
    if best is None:
        return np.zeros_like(correct), NEVER, 0.0, False
    _, _, h, mm = best
    return (score >= h) & (margin >= mm), h, mm, True


def _fit_floor(
    score: NDArray[np.float64],
    has_pair: NDArray[np.bool_],
    rest: NDArray[np.bool_],
    cfg: ThresholdsConfig,
) -> tuple[float, bool]:
    """Largest no_match zone (lowest scores) in which enough rows truly have no pair."""
    idx = np.flatnonzero(rest)
    order = idx[np.argsort(score[idx], kind="stable")]
    s_sorted = score[order]
    no_pair = ~has_pair[order]
    cum = np.cumsum(no_pair)
    best_n = 0
    for n in range(cfg.min_no_match_support, len(order) + 1):
        if n < len(order) and s_sorted[n] == s_sorted[n - 1]:
            continue  # a cut must fall between two different scores
        if cum[n - 1] / n >= cfg.no_match_target_precision:
            best_n = n
    if best_n == 0:
        return -NEVER, False
    if best_n == len(order):
        return float(s_sorted[-1]) + 1e-6, True
    return float((s_sorted[best_n - 1] + s_sorted[best_n]) / 2), True


def _fit_low(
    score: NDArray[np.float64],
    answerable: NDArray[np.bool_],
    rest: NDArray[np.bool_],
    floor: float,
    cfg: ThresholdsConfig,
) -> tuple[float, bool]:
    """Widest arbiter band in which enough rows are answerable."""
    cand_mask = rest & (score >= floor)
    best: float | None = None
    for low in sorted({float(s) for s in score[cand_mask]}):
        band = cand_mask & (score >= low)
        n = int(band.sum())
        if n >= cfg.min_llm_support and answerable[band].mean() >= cfg.llm_min_answerable:
            best = low
            break  # ascending: the first feasible low gives the widest band
    return (NEVER, False) if best is None else (best, True)
