"""The central trade-off: target precision of the auto zone vs how much gets automated."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from entity_cascade.config import ThresholdsConfig
from entity_cascade.data.split import TestSplit, ValSplit
from entity_cascade.eval.types import NanFloat
from entity_cascade.eval.zone_metrics import system_metrics, zone_decisions
from entity_cascade.rerank.ranker import RowRanking
from entity_cascade.schemas import Zone
from entity_cascade.zones.assign import Thresholds
from entity_cascade.zones.thresholds import fit_thresholds


class TradeoffRow(BaseModel):
    label: str
    target_precision: NanFloat
    wilson_z: NanFloat
    feasible: bool
    high: NanFloat
    val_auto_share: NanFloat
    val_auto_precision: NanFloat
    test_auto_share: NanFloat
    test_auto_precision: NanFloat
    test_human_share: NanFloat  # cascade without the LLM: everything not auto/no_match
    test_system_precision: NanFloat
    test_system_recall: NanFloat


def _auto_stats(
    ids: Sequence[str],
    rankings: Mapping[str, RowRanking],
    truth: Mapping[str, frozenset[str]],
    t: Thresholds,
) -> tuple[float, float]:
    decisions = zone_decisions(ids, rankings, t)
    auto = [i for i, d in decisions.items() if d.zone is Zone.AUTO_MATCH]
    share = len(auto) / len(ids) if ids else float("nan")
    prec = sum(rankings[i].top1_id in truth[i] for i in auto) / len(auto) if auto else float("nan")
    return share, prec


def tradeoff_rows(
    val: ValSplit,
    test: TestSplit,
    rankings_val: Mapping[str, RowRanking],
    rankings_test: Mapping[str, RowRanking],
    base: ThresholdsConfig,
    settings: Sequence[tuple[float, float]],
) -> list[TradeoffRow]:
    """For each (target precision, wilson z): fit thresholds on val, report val and test."""
    val_ids = [r.id for r in val.rows]
    test_ids = [r.id for r in test.rows]
    out: list[TradeoffRow] = []
    for target, z in settings:
        cfg = base.model_copy(update={"target_precision": target, "wilson_z": z})
        t = fit_thresholds(val, rankings_val, cfg)  # val only
        v_share, v_prec = _auto_stats(val_ids, rankings_val, val.truth, t)
        t_share, t_prec = _auto_stats(test_ids, rankings_test, test.truth, t)
        sysm = system_metrics(zone_decisions(test_ids, rankings_test, t), test.truth)
        label = f"{target:.1%}" + (" (Wilson 95% bound)" if z > 0 else "")
        out.append(
            TradeoffRow(
                label=label,
                target_precision=target,
                wilson_z=z,
                feasible=t.auto_feasible,
                high=t.high,
                val_auto_share=v_share,
                val_auto_precision=v_prec,
                test_auto_share=t_share,
                test_auto_precision=t_prec,
                test_human_share=sysm.human_share,
                test_system_precision=sysm.precision,
                test_system_recall=sysm.recall,
            )
        )
    return out
