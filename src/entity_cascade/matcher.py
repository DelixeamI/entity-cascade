"""Run the whole cascade on arbitrary price-list rows against a catalog (no labels needed)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from entity_cascade.calibration import Calibration
from entity_cascade.config import AppConfig
from entity_cascade.llm.arbiter import ArbiterItem, ArbiterResult, arbitrate, is_resolved
from entity_cascade.llm.base import CassetteMiss, LLMClient
from entity_cascade.pipeline import compute_ranking_inputs
from entity_cascade.rerank.cross_encoder import Reranker
from entity_cascade.rerank.ranker import RowRanking, ScoredCandidate, rank_row
from entity_cascade.retrieval.candidates import CandidateGenerator
from entity_cascade.schemas import Record, Zone

Verdict = Literal["match", "no_match", "needs_human"]


@dataclass(frozen=True)
class MatchOutcome:
    row: Record
    zone: Zone
    verdict: Verdict
    handled_by: Literal["auto", "llm", "human"]
    candidate: Record | None  # the matched catalog item, or the best guess for a human to check
    score: float
    margin: float
    reason: str
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0


def _feature_note(c: ScoredCandidate) -> str:
    f = c.features
    notes = []
    if f.code_exact:
        notes.append("model code equal")
    elif f.code_prefix:
        notes.append("model code is a prefix match")
    elif f.code_conflict:
        notes.append("model codes differ")
    if f.brand_match:
        notes.append("brand equal")
    elif f.brand_conflict:
        notes.append("brand differs")
    return ", ".join(notes) if notes else "no model code or brand evidence"


def _zone_reason(zone: Zone, rk: RowRanking, cal: Calibration) -> str:
    t = cal.thresholds
    base = f"score {rk.score:.2f}, margin {rk.margin:.2f}"
    if not rk.candidates:
        return "no candidates retrieved"
    note = _feature_note(rk.candidates[0])
    if zone is Zone.AUTO_MATCH:
        return f"{base} (needs >= {t.high:.2f} and >= {t.margin_min:.2f}); {note}"
    if zone is Zone.NO_MATCH:
        return f"{base} is below the no-match floor {t.floor:.2f}"
    if zone is Zone.LLM_ARBITER:
        return f"{base} is below the auto thresholds or the margin is thin; {note}"
    return f"{base} is between the no-match floor and the arbiter band; {note}"


def match_rows(
    cfg: AppConfig,
    cal: Calibration,
    rows: Sequence[Record],
    catalog: Sequence[Record],
    gen: CandidateGenerator,
    reranker: Reranker,
    llm: LLMClient | None = None,
) -> tuple[list[MatchOutcome], str | None]:
    """Returns outcomes in input order and a note if the LLM stage was degraded."""
    from entity_cascade.zones.assign import assign_zone

    by_id = {c.id: c for c in catalog}
    inputs = compute_ranking_inputs(cfg, gen, catalog, rows, reranker)
    rankings = {r.id: rank_row(inputs.raws[r.id], cal.weights) for r in rows}
    zones = {r.id: assign_zone(rankings[r.id], cal.thresholds) for r in rows}

    results: dict[str, ArbiterResult] = {}
    note: str | None = None
    gray = [r for r in rows if zones[r.id] is Zone.LLM_ARBITER]
    if llm is not None and gray:
        items = [
            ArbiterItem(
                r,
                tuple(
                    by_id[c.catalog_id] for c in rankings[r.id].candidates[: cfg.llm.top_candidates]
                ),
            )
            for r in gray
        ]
        try:
            results = {res.row_id: res for res in asyncio.run(arbitrate(llm, items, cfg.llm))}
        except CassetteMiss as exc:
            note = f"LLM stage skipped, gray-zone rows go to a human: {exc}"

    out: list[MatchOutcome] = []
    for row in rows:
        rk, zone = rankings[row.id], zones[row.id]
        top = by_id[rk.top1_id] if rk.top1_id else None
        reason = _zone_reason(zone, rk, cal)
        res = results.get(row.id)

        def outcome(
            verdict: Verdict,
            handled_by: Literal["auto", "llm", "human"],
            candidate: Record | None,
            why: str,
            row: Record = row,
            rk: RowRanking = rk,
            zone: Zone = zone,
            res: ArbiterResult | None = res,
        ) -> MatchOutcome:
            return MatchOutcome(
                row=row,
                zone=zone,
                verdict=verdict,
                handled_by=handled_by,
                candidate=candidate,
                score=rk.score,
                margin=rk.margin,
                reason=why,
                llm_input_tokens=res.input_tokens if res else 0,
                llm_output_tokens=res.output_tokens if res else 0,
            )

        if zone is Zone.AUTO_MATCH:
            out.append(outcome("match", "auto", top, "auto_match: " + reason))
        elif zone is Zone.NO_MATCH:
            out.append(outcome("no_match", "auto", None, "no_match: " + reason))
        elif (
            res is not None and res.answer is not None and is_resolved(res, cfg.llm.min_confidence)
        ):
            chosen = by_id[res.chosen_id] if res.chosen_id else None
            why = (
                f"llm_arbiter (confidence {res.answer.confidence:.2f}): "
                f"{res.answer.reason or 'no reason given'}"
            )
            out.append(outcome("match" if chosen else "no_match", "llm", chosen, why))
        else:
            why = zone.value + ": " + reason
            if res is not None:
                why += (
                    f"; the LLM answer was not accepted ({res.error})"
                    if res.answer is None
                    else f"; the LLM was not confident enough ({res.answer.confidence:.2f})"
                )
            elif zone is Zone.LLM_ARBITER:
                why += "; no LLM was used"
            out.append(outcome("needs_human", "human", top, why))
    return out, note
