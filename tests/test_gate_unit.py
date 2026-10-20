"""The gate function itself, on hand-built results."""

from __future__ import annotations

from entity_cascade.calibration import Calibration
from entity_cascade.data.stats import DatasetStats
from entity_cascade.eval.gate import GateThresholds, check_gate
from entity_cascade.eval.report import (
    EvalResults,
    RecallRow,
    SystemRow,
    Timings,
    Top1Row,
    ZoneBlock,
)
from entity_cascade.eval.zone_metrics import SystemMetrics, ZoneRow
from entity_cascade.rerank.features import FeatureWeights
from entity_cascade.zones.assign import Thresholds

GATE = GateThresholds(
    candidate_recall_at_20_min=0.95,
    top1_reranker_features_min=0.8,
    auto_zone_precision_min=0.95,
    auto_zone_share_min=0.3,
    system_precision_min=0.9,
    system_recall_min=0.9,
    human_share_max=0.5,
)


def results(
    *,
    recall: float = 1.0,
    top1: float = 0.9,
    auto_prec: float = 1.0,
    auto_share: float = 0.6,
    precision: float = 0.95,
    rec: float = 0.95,
    human: float = 0.2,
    with_llm: bool = True,
) -> EvalResults:
    stats = DatasetStats(
        left_rows=1,
        right_rows=1,
        pairs=1,
        left_with_pair=1,
        left_without_pair=0,
        left_with_multiple_pairs=0,
        right_with_multiple_left=0,
        left_with_brand=1,
        left_with_code=1,
        pairs_sharing_code=1,
    )
    sysm = SystemMetrics(
        rows=10,
        automation_rate=0.8,
        llm_share=0.2,
        human_share=human,
        automated_match_precision=0.9,
        automated_accuracy=0.9,
        precision=precision,
        recall=rec,
        automated_matches=5,
        automated_wrong=1,
    )
    zone = ZoneRow(
        zone="auto_match", rows=6, share=auto_share, top1_correct=auto_prec, without_pair=0.0
    )
    return EvalResults(
        config_path="x",
        dataset_name="x",
        seed=1,
        embedding_model="e",
        reranker_model="r",
        dataset_stats=stats,
        catalog_size=1,
        new_product_fraction=0.0,
        splits=[],
        candidate_recall=[RecallRow(mode="hybrid", split="test", n_rows=1, recall={20: recall})],
        modes=[],
        top1=[Top1Row(mode="hybrid + reranker + features", split="test", n_rows=1, accuracy=top1)],
        calibration=Calibration(weights=FeatureWeights(), thresholds=Thresholds()),
        target_precision=0.98,
        zone_blocks=[ZoneBlock(split="test", zones=[zone])],
        systems=[SystemRow(split="test", name="cascade + LLM arbiter", metrics=sysm)]
        if with_llm
        else [],
        tradeoff=[],
        llm=None,
        llm_note="no cassettes",
        timings=Timings(index_build_s=0, retrieval_ms_per_row=0, rerank_ms_per_row=0, bench_rows=0),
    )


def test_gate_passes_when_everything_is_above_the_floors() -> None:
    assert check_gate(results(), GATE) == []


def test_gate_reports_each_violation() -> None:
    fails = check_gate(
        results(
            recall=0.5, top1=0.1, auto_prec=0.5, auto_share=0.0, precision=0.1, rec=0.1, human=0.9
        ),
        GATE,
    )
    assert len(fails) == 7
    assert any("recall@20" in f for f in fails)
    assert any("human share" in f and ">" in f for f in fails)


def test_gate_fails_without_llm_run_when_required() -> None:
    fails = check_gate(results(with_llm=False), GATE)
    assert fails and "LLM arbiter did not run" in fails[0]
    relaxed = GATE.model_copy(update={"require_llm_run": False})
    assert check_gate(results(with_llm=False), relaxed) == []


def test_gate_treats_nan_as_failure() -> None:
    assert check_gate(results(auto_prec=float("nan")), GATE)
