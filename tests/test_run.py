"""Orchestration on the fixture with fake models: plumbing, report text, record/replay."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from entity_cascade.config import PROJECT_ROOT, AppConfig, load_config
from entity_cascade.eval.report import render_markdown
from entity_cascade.eval.run import (
    BENCH_ROWS,
    CascadeRun,
    Models,
    _run_arbiter,
    evaluate,
    llm_items,
    make_llm,
    run_cascade,
)
from entity_cascade.llm.arbiter import ArbiterItem
from entity_cascade.llm.base import LLMClient
from entity_cascade.llm.fake import HeuristicFakeLLM
from entity_cascade.llm.recorded import RecordedLLM, RecordingLLM
from entity_cascade.schemas import Zone
from helpers import HashingEmbedder


class OverlapReranker:
    """Fake cross-encoder: shared-token count, scaled so scores spread over a useful range."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [4.0 * len(set(q.lower().split()) & set(d.lower().split())) - 3.0 for q, d in pairs]


def fixture_cfg(tmp_path: Path) -> AppConfig:
    cfg = load_config(PROJECT_ROOT / "configs" / "fixture.yaml")
    return cfg.model_copy(
        update={
            "llm": cfg.llm.model_copy(
                update={
                    "cassette_dir": tmp_path / "cassettes",
                    "model": "fake-heuristic-v1",
                    "price_input_per_1m": 1.0,
                    "price_output_per_1m": 2.0,
                }
            ),
            "rerank": cfg.rerank.model_copy(update={"cache_path": None}),
        }
    )


@pytest.fixture(scope="module")
def run_and_cfg(tmp_path_factory: pytest.TempPathFactory) -> tuple[CascadeRun, AppConfig]:
    cfg = fixture_cfg(tmp_path_factory.mktemp("run"))
    models = Models(HashingEmbedder(), OverlapReranker(), OverlapReranker())
    return run_cascade(cfg, models), cfg


def test_run_cascade_covers_every_row_once(run_and_cfg: tuple[CascadeRun, AppConfig]) -> None:
    run, _ = run_and_cfg
    assert set(run.decisions["val"]) == {r.id for r in run.splits.val.rows}
    assert set(run.decisions["test"]) == {r.id for r in run.splits.test.rows}
    assert run.bench_rows == min(BENCH_ROWS, len(run.splits.val.rows))
    assert run.cal.thresholds.floor <= run.cal.thresholds.low <= run.cal.thresholds.high


def test_llm_items_are_exactly_the_arbiter_zone(run_and_cfg: tuple[CascadeRun, AppConfig]) -> None:
    run, cfg = run_and_cfg
    items = llm_items(run, "test")
    zone_rows = {i for i, d in run.decisions["test"].items() if d.zone is Zone.LLM_ARBITER}
    assert {it.row.id for it in items} == zone_rows
    for it in items:
        assert 1 <= len(it.candidates) <= cfg.llm.top_candidates
        ranked = run.rankings["test"][it.row.id].candidates
        assert [c.id for c in it.candidates] == [c.catalog_id for c in ranked[: len(it.candidates)]]


def test_evaluate_without_cassettes_reports_llm_as_not_run(
    run_and_cfg: tuple[CascadeRun, AppConfig],
) -> None:
    run, cfg = run_and_cfg
    items = llm_items(run, "test")
    if not items:
        pytest.skip("no gray-zone rows in this fixture run")
    results = evaluate(run, RecordedLLM(cfg.llm.model, cfg.llm.cassette_dir), "fixture.yaml")
    assert results.llm is None
    assert results.llm_note and "no cassette" in results.llm_note
    md = render_markdown(results)
    assert "Not run: no recorded LLM answers" in md and "make record" in md
    assert "cascade + LLM arbiter" not in md


def test_record_then_replay_gives_identical_numbers(
    run_and_cfg: tuple[CascadeRun, AppConfig],
) -> None:
    import asyncio

    run, cfg = run_and_cfg
    items = llm_items(run, "test")
    if not items:
        pytest.skip("no gray-zone rows in this fixture run")

    recorder = RecordingLLM(HeuristicFakeLLM(), cfg.llm.cassette_dir)
    asyncio.run(_run_arbiter(recorder, items, cfg))
    assert len(list(cfg.llm.cassette_dir.glob("*.json"))) == len({_prompt_key(i) for i in items})

    replayed = evaluate(run, make_llm(cfg, "recorded"), "fixture.yaml")
    live = evaluate(run, HeuristicFakeLLM(), "fixture.yaml")
    assert replayed.llm is not None and live.llm is not None
    a, b = replayed.llm.metrics, live.llm.metrics
    assert (a.resolved, a.escalated, a.resolved_accuracy) == (
        b.resolved,
        b.escalated,
        b.resolved_accuracy,
    )
    assert (a.input_tokens, a.output_tokens) == (b.input_tokens, b.output_tokens)
    assert a.cost_usd_run == pytest.approx(a.input_tokens * 1.0 / 1e6 + a.output_tokens * 2.0 / 1e6)

    md = render_markdown(replayed)
    assert "is a test double, not an LLM" in md
    assert "cascade + LLM arbiter" in md


def _prompt_key(item: ArbiterItem) -> str:
    from entity_cascade.llm.arbiter import build_prompt

    return build_prompt(item.row, item.candidates)


def test_system_rows_are_consistent(run_and_cfg: tuple[CascadeRun, AppConfig]) -> None:
    run, cfg = run_and_cfg
    results = evaluate(run, None, "fixture.yaml")
    by = {(s.split, s.name): s.metrics for s in results.systems}
    everything = by[("test", "everything to a human")]
    assert (
        everything.human_share == 1.0 and everything.recall == 1.0 and everything.precision == 1.0
    )
    no_cascade = by[("test", "accept every top-1 (no cascade)")]
    assert no_cascade.automation_rate == 1.0
    cascade = by[("test", "cascade, llm_arbiter rows go to a human")]
    assert cascade.automation_rate + cascade.human_share == pytest.approx(1.0)
    assert cfg.llm.provider == "recorded"


def test_make_llm_kinds(tmp_path: Path) -> None:
    cfg = fixture_cfg(tmp_path)
    assert isinstance(make_llm(cfg, "recorded"), RecordedLLM)
    assert make_llm(cfg, "fake").model == "fake-heuristic-v1"
    with pytest.raises(ValueError, match="unknown LLM provider"):
        make_llm(cfg, "nope")
    client: LLMClient = make_llm(cfg, "recorded")
    assert client.model == "fake-heuristic-v1"
