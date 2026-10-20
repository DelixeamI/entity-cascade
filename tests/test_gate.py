"""CI gate: real models on the fixture slice, RecordedLLM cassettes, eval_thresholds.yaml limits.

The cassettes in tests/cassettes/ were recorded with the heuristic test double (model
"fake-heuristic-v1"), not a real LLM, so this gate protects the pipeline and its plumbing
against regressions. It says nothing about LLM quality. Re-record with a real provider
(`make record`) and tighten eval_thresholds.yaml when you want it to.
"""

from __future__ import annotations

import pytest

from entity_cascade.config import PROJECT_ROOT, load_config
from entity_cascade.eval.gate import check_gate, load_gate
from entity_cascade.eval.run import build_models, evaluate, make_llm, run_cascade

pytestmark = pytest.mark.slow


def test_fixture_meets_eval_thresholds() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "fixture.yaml")
    cfg = cfg.model_copy(update={"rerank": cfg.rerank.model_copy(update={"cache_path": None})})
    run = run_cascade(cfg, build_models(cfg))
    results = evaluate(run, make_llm(cfg, "recorded"), "configs/fixture.yaml")
    gate = load_gate(PROJECT_ROOT / "eval_thresholds.yaml")
    failures = check_gate(results, gate)
    assert not failures, "gate failed:\n" + "\n".join(failures)
