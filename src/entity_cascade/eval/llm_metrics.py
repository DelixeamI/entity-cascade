"""Arbiter metrics: gray-zone rows taken off the human queue, how precisely, at what cost."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from entity_cascade.config import LLMConfig
from entity_cascade.eval.types import NanFloat
from entity_cascade.eval.zone_metrics import Decision
from entity_cascade.llm.arbiter import ArbiterResult, is_resolved
from entity_cascade.rerank.ranker import RowRanking


class ArbiterMetrics(BaseModel):
    rows_in_zone: int
    resolved: int  # answered with confidence >= min_confidence: no human needed
    escalated: int  # provider error, unusable output or low confidence: goes to a human
    errors: int  # subset of escalated: provider failure or unusable output
    resolved_share: NanFloat  # resolved / rows_in_zone: the human workload removed
    resolved_accuracy: NanFloat  # resolved answers that are right (a match or "none")
    resolved_match_precision: NanFloat  # resolved "this item" answers that are right
    resolved_none_precision: NanFloat  # resolved "none" answers on rows that truly have no pair
    top1_accuracy_same_rows: NanFloat  # what blindly taking the cascade's top-1 would score
    true_item_shown_share: NanFloat  # rows with a pair where a true item is among the shown ones
    calls: int
    input_tokens: int
    output_tokens: int
    mean_latency_s: NanFloat
    cost_usd_run: NanFloat
    cost_usd_per_1000_rows: NanFloat  # run cost scaled to 1000 price-list rows (all zones)
    rows_in_run: int
    min_confidence: NanFloat


def _answer_ok(res: ArbiterResult, truth: frozenset[str]) -> bool:
    chosen = res.chosen_id
    return (chosen in truth) if chosen is not None else not truth


def cost_usd(input_tokens: int, output_tokens: int, cfg: LLMConfig) -> float:
    return (
        input_tokens * cfg.price_input_per_1m + output_tokens * cfg.price_output_per_1m
    ) / 1_000_000


def arbiter_metrics(
    results: Sequence[ArbiterResult],
    truth: Mapping[str, frozenset[str]],
    rankings: Mapping[str, RowRanking],
    cfg: LLMConfig,
    rows_in_run: int,
) -> ArbiterMetrics:
    nan = float("nan")
    n = len(results)
    resolved = [r for r in results if is_resolved(r, cfg.min_confidence)]
    matches = [r for r in resolved if r.chosen_id is not None]
    nones = [r for r in resolved if r.chosen_id is None]
    with_pair = [r for r in results if truth[r.row_id]]
    in_tok = sum(r.input_tokens for r in results)
    out_tok = sum(r.output_tokens for r in results)
    run_cost = cost_usd(in_tok, out_tok, cfg)
    return ArbiterMetrics(
        rows_in_zone=n,
        resolved=len(resolved),
        escalated=n - len(resolved),
        errors=sum(1 for r in results if r.answer is None),
        resolved_share=len(resolved) / n if n else nan,
        resolved_accuracy=(
            sum(_answer_ok(r, truth[r.row_id]) for r in resolved) / len(resolved)
            if resolved
            else nan
        ),
        resolved_match_precision=(
            sum(_answer_ok(r, truth[r.row_id]) for r in matches) / len(matches) if matches else nan
        ),
        resolved_none_precision=(
            sum(_answer_ok(r, truth[r.row_id]) for r in nones) / len(nones) if nones else nan
        ),
        top1_accuracy_same_rows=(
            sum(rankings[r.row_id].top1_id in truth[r.row_id] for r in results) / n if n else nan
        ),
        true_item_shown_share=(
            sum(bool(truth[r.row_id] & set(r.shown_ids)) for r in with_pair) / len(with_pair)
            if with_pair
            else nan
        ),
        calls=n,
        input_tokens=in_tok,
        output_tokens=out_tok,
        mean_latency_s=sum(r.latency_s for r in results) / n if n else nan,
        cost_usd_run=run_cost,
        cost_usd_per_1000_rows=run_cost / rows_in_run * 1000 if rows_in_run else nan,
        rows_in_run=rows_in_run,
        min_confidence=cfg.min_confidence,
    )


def apply_arbiter(
    decisions: Mapping[str, Decision], results: Sequence[ArbiterResult], min_confidence: float
) -> dict[str, Decision]:
    """Rows the arbiter resolved are answered by it; the others stay with a human."""
    out = dict(decisions)
    for res in results:
        if is_resolved(res, min_confidence):
            base = decisions[res.row_id]
            out[res.row_id] = Decision(res.row_id, base.zone, res.chosen_id, "llm")
    return out
