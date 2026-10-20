"""Choose the arbiter's prompt version and min_confidence on val. Takes val data only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from entity_cascade.config import LLMConfig
from entity_cascade.data.split import ValSplit
from entity_cascade.eval.llm_metrics import apply_arbiter
from entity_cascade.eval.types import NanFloat
from entity_cascade.eval.zone_metrics import Decision, system_metrics
from entity_cascade.llm.arbiter import PROMPT_VERSIONS, ArbiterResult, is_resolved

CONFIDENCE_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


class TuningCell(BaseModel):
    version: str
    min_confidence: float
    resolved: int  # gray-zone rows the arbiter would answer at this threshold
    resolved_share: NanFloat
    resolved_accuracy: (
        NanFloat  # right answers among the resolved ("none" is right on no-pair rows)
    )
    match_precision: NanFloat  # resolved "this item" answers that are right
    none_precision: NanFloat  # resolved "none" answers that are right
    system_precision: NanFloat  # val, cascade + arbiter, a human answers the rest
    system_recall: NanFloat
    human_share: NanFloat


class TuningResult(BaseModel):
    rows_in_zone: int  # val rows sent to the arbiter
    target_accuracy: float
    min_resolved_support: int
    cells: list[TuningCell]
    chosen_version: str | None
    chosen_min_confidence: float | None
    feasible: bool  # False: no cell reached the target accuracy; the most accurate cell is chosen


def _right(res: ArbiterResult, truth: frozenset[str]) -> bool:
    chosen = res.chosen_id
    return (chosen in truth) if chosen is not None else not truth


def grid_cells(
    version: str,
    results: Sequence[ArbiterResult],
    val: ValSplit,
    decisions_val: Mapping[str, Decision],
) -> list[TuningCell]:
    nan = float("nan")
    n = len(results)
    cells: list[TuningCell] = []
    for t in CONFIDENCE_GRID:
        res = [r for r in results if is_resolved(r, t)]
        matches = [r for r in res if r.chosen_id is not None]
        nones = [r for r in res if r.chosen_id is None]

        def ok(rs: list[ArbiterResult]) -> int:
            return sum(_right(r, val.truth[r.row_id]) for r in rs)

        sysm = system_metrics(apply_arbiter(decisions_val, results, t), val.truth)
        cells.append(
            TuningCell(
                version=version,
                min_confidence=t,
                resolved=len(res),
                resolved_share=len(res) / n if n else nan,
                resolved_accuracy=ok(res) / len(res) if res else nan,
                match_precision=ok(matches) / len(matches) if matches else nan,
                none_precision=ok(nones) / len(nones) if nones else nan,
                system_precision=sysm.precision,
                system_recall=sysm.recall,
                human_share=sysm.human_share,
            )
        )
    return cells


def select(
    cells: Sequence[TuningCell], target: float, min_support: int
) -> tuple[TuningCell | None, bool]:
    """Widest feasible setting; if none reaches the target, the most accurate one (feasible=False).

    Feasible: at least `min_support` resolved rows and accuracy >= target. Ties go to higher
    accuracy, then to the earlier prompt version, then to the higher confidence threshold.
    """
    order = {v: i for i, v in enumerate(PROMPT_VERSIONS)}
    supported = [
        c for c in cells if c.resolved >= min_support and c.resolved_accuracy == c.resolved_accuracy
    ]
    feasible = [c for c in supported if c.resolved_accuracy >= target]
    if feasible:
        best = min(
            feasible,
            key=lambda c: (
                -c.resolved,
                -c.resolved_accuracy,
                order.get(c.version, 99),
                -c.min_confidence,
            ),
        )
        return best, True
    if supported:
        best = min(
            supported,
            key=lambda c: (
                -c.resolved_accuracy,
                -c.resolved,
                order.get(c.version, 99),
                -c.min_confidence,
            ),
        )
        return best, False
    return None, False


def tune(
    results_by_version: Mapping[str, Sequence[ArbiterResult]],
    val: ValSplit,
    decisions_val: Mapping[str, Decision],
    cfg: LLMConfig,
) -> TuningResult:
    cells: list[TuningCell] = []
    rows = 0
    for version, results in results_by_version.items():
        rows = max(rows, len(results))
        cells += grid_cells(version, results, val, decisions_val)
    best, feasible = select(cells, cfg.target_accuracy, cfg.min_resolved_support)
    return TuningResult(
        rows_in_zone=rows,
        target_accuracy=cfg.target_accuracy,
        min_resolved_support=cfg.min_resolved_support,
        cells=cells,
        chosen_version=best.version if best else None,
        chosen_min_confidence=best.min_confidence if best else None,
        feasible=feasible,
    )
