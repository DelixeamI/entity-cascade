"""Zone table and end-to-end metrics under the "a human is always right" assumption."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from entity_cascade.eval.types import NanFloat
from entity_cascade.rerank.ranker import RowRanking
from entity_cascade.schemas import Zone
from entity_cascade.zones.assign import Thresholds, assign_zone

HandledBy = Literal["auto", "llm", "human"]


@dataclass(frozen=True)
class Decision:
    row_id: str
    zone: Zone
    pred_id: str | None  # system answer; for handled_by == "human" it is not used
    handled_by: HandledBy


class ZoneRow(BaseModel):
    zone: str
    rows: int
    share: NanFloat
    top1_correct: NanFloat  # share of rows whose top-1 is a true item
    without_pair: NanFloat  # share of rows that truly have no pair


class SystemMetrics(BaseModel):
    """Human answers are assumed correct. See DESIGN.md for the definitions."""

    rows: int
    automation_rate: NanFloat  # rows answered without a human (rules + LLM)
    llm_share: NanFloat  # of which: rows answered by the LLM arbiter
    human_share: NanFloat
    automated_match_precision: NanFloat  # automated "this item" answers that are right
    automated_accuracy: NanFloat  # all automated answers (match or none) that are right
    precision: NanFloat  # system-wide: TP / predicted matches, humans counted as right
    recall: NanFloat  # system-wide: TP / rows with a true pair
    automated_matches: int
    automated_wrong: int


def zone_decisions(
    rows: Sequence[str], rankings: Mapping[str, RowRanking], t: Thresholds
) -> dict[str, Decision]:
    """Decisions from the zones alone: llm_arbiter rows are not yet answered (-> human)."""
    out: dict[str, Decision] = {}
    for rid in rows:
        zone = assign_zone(rankings[rid], t)
        if zone is Zone.AUTO_MATCH:
            out[rid] = Decision(rid, zone, rankings[rid].top1_id, "auto")
        elif zone is Zone.NO_MATCH:
            out[rid] = Decision(rid, zone, None, "auto")
        else:
            out[rid] = Decision(rid, zone, None, "human")
    return out


def zone_table(
    zones: Mapping[str, Zone],
    rankings: Mapping[str, RowRanking],
    truth: Mapping[str, frozenset[str]],
) -> list[ZoneRow]:
    n = len(zones)
    out: list[ZoneRow] = []
    for zone in Zone:
        ids = [rid for rid, z in zones.items() if z is zone]
        k = len(ids)
        correct = sum(1 for rid in ids if rankings[rid].top1_id in truth[rid])
        no_pair = sum(1 for rid in ids if not truth[rid])
        out.append(
            ZoneRow(
                zone=zone.value,
                rows=k,
                share=k / n if n else float("nan"),
                top1_correct=correct / k if k else float("nan"),
                without_pair=no_pair / k if k else float("nan"),
            )
        )
    return out


def system_metrics(
    decisions: Mapping[str, Decision], truth: Mapping[str, frozenset[str]]
) -> SystemMetrics:
    n = len(decisions)
    auto_match = auto_match_ok = auto_ok = auto_n = human = tp_human = llm_n = 0
    for rid, d in decisions.items():
        t = truth[rid]
        llm_n += d.handled_by == "llm"
        if d.handled_by == "human":
            human += 1
            tp_human += bool(t)
            continue
        auto_n += 1
        if d.pred_id is not None:
            auto_match += 1
            ok = d.pred_id in t
            auto_match_ok += ok
        else:
            ok = not t
        auto_ok += ok
    with_pair = sum(1 for t in truth.values() if t)
    tp = auto_match_ok + tp_human
    predicted = auto_match + tp_human
    nan = float("nan")
    return SystemMetrics(
        rows=n,
        automation_rate=auto_n / n if n else nan,
        llm_share=llm_n / n if n else nan,
        human_share=human / n if n else nan,
        automated_match_precision=auto_match_ok / auto_match if auto_match else nan,
        automated_accuracy=auto_ok / auto_n if auto_n else nan,
        precision=tp / predicted if predicted else nan,
        recall=tp / with_pair if with_pair else nan,
        automated_matches=auto_match,
        automated_wrong=auto_n - auto_ok,
    )
