"""CI gate: the fixture slice must keep meeting the floors in eval_thresholds.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from entity_cascade.eval.report import EvalResults


class GateThresholds(BaseModel):
    """Floors and ceilings with headroom below the measured values, so noise does not trip them
    but a real regression does. Raise them when the pipeline genuinely improves."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_recall_at_20_min: float
    top1_reranker_features_min: float  # hybrid + reranker + features, test
    auto_zone_precision_min: float  # test
    auto_zone_share_min: float  # test
    system_precision_min: float  # test, cascade + LLM arbiter, human always right
    system_recall_min: float
    human_share_max: float
    require_llm_run: bool = True


def load_gate(path: Path) -> GateThresholds:
    with path.open(encoding="utf-8") as f:
        return GateThresholds.model_validate(yaml.safe_load(f))


def check_gate(r: EvalResults, g: GateThresholds) -> list[str]:
    """Returns the list of violated limits; empty means the gate passes."""
    fails: list[str] = []

    def floor(name: str, value: float, limit: float) -> None:
        if not value >= limit:  # also catches NaN
            fails.append(f"{name}: {value:.4f} < {limit:.4f}")

    def ceil(name: str, value: float, limit: float) -> None:
        if not value <= limit:
            fails.append(f"{name}: {value:.4f} > {limit:.4f}")

    recall = next(x for x in r.candidate_recall if x.mode == "hybrid" and x.split == "test")
    floor("hybrid candidate recall@20 (test)", recall.recall[20], g.candidate_recall_at_20_min)
    top1 = next(t for t in r.top1 if t.mode == "hybrid + reranker + features" and t.split == "test")
    floor("top-1 accuracy, reranker + features (test)", top1.accuracy, g.top1_reranker_features_min)

    zones = next(z for z in r.zone_blocks if z.split == "test")
    auto = next(z for z in zones.zones if z.zone == "auto_match")
    floor("auto_match share (test)", auto.share, g.auto_zone_share_min)
    floor("auto_match precision (test)", auto.top1_correct, g.auto_zone_precision_min)

    llm_row = next((s for s in r.systems if s.name == "cascade + LLM arbiter"), None)
    if llm_row is None:
        if g.require_llm_run:
            fails.append(f"LLM arbiter did not run: {r.llm_note}")
        return fails
    floor("system precision (test, with LLM)", llm_row.metrics.precision, g.system_precision_min)
    floor("system recall (test, with LLM)", llm_row.metrics.recall, g.system_recall_min)
    ceil("human share (test, with LLM)", llm_row.metrics.human_share, g.human_share_max)
    return fails
