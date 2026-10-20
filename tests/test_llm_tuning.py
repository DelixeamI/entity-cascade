"""Prompt versions and the val-only selection of prompt version + min_confidence."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from entity_cascade.config import PROJECT_ROOT, LLMConfig
from entity_cascade.data.split import ValSplit
from entity_cascade.eval.llm_tuning import CONFIDENCE_GRID, grid_cells, select, tune
from entity_cascade.eval.zone_metrics import Decision
from entity_cascade.llm.arbiter import PROMPT_VERSIONS, ArbiterResult, build_prompt, parse_answer
from entity_cascade.schemas import ArbiterAnswer, Record, Zone


def rec(rid: str, name: str, **fields: str) -> Record:
    return Record(id=rid, name=name, brand=fields.get("manufacturer"), fields=fields)


# ---- prompt versions ----


def test_v1_text_is_frozen() -> None:
    """Recorded v1 cassettes stay valid only while v1 does not change."""
    p = build_prompt(rec("r", "Sony PSLX350H"), [rec("a", "Sony PSLX350H")], "v1")
    assert p == build_prompt(rec("r", "Sony PSLX350H"), [rec("a", "Sony PSLX350H")])
    assert "The same product has the same brand and the same model number." in p
    assert "model number (letters" not in p


def test_all_versions_build_and_differ() -> None:
    row, cands = rec("r", "Sony PSLX350H turntable"), [rec("a", "Sony PSLX350H"), rec("b", "x")]
    prompts = {v: build_prompt(row, cands, v) for v in PROMPT_VERSIONS}
    assert len(set(prompts.values())) == len(PROMPT_VERSIONS)
    for p in prompts.values():
        assert "Price-list item:\nname: Sony PSLX350H turntable" in p
        assert "\n1. name: Sony PSLX350H" in p and "\n2. name: x" in p


def test_v4_adds_rule_based_checks_only_there() -> None:
    row = rec("r", "Sony Turntable - PSLX350H", manufacturer="Sony")
    cands = [
        rec("a", "Sony PSLX350H", manufacturer="Sony"),
        rec("b", "Canon S210", manufacturer="Canon"),
    ]
    v3, v4 = build_prompt(row, cands, "v3"), build_prompt(row, cands, "v4")
    assert "check:" not in v3
    assert "1. name: Sony PSLX350H; brand: Sony; check: model number equal, brand equal" in v4
    assert "check: model numbers differ, brand differs" in v4


def test_v3_asks_for_reason_first_and_parser_accepts_any_key_order() -> None:
    assert '{"reason": "<comparison>", "choice"' in build_prompt(
        rec("r", "x"), [rec("a", "y")], "v3"
    )
    a = parse_answer('{"reason": "same model", "choice": 1, "confidence": 0.9}', 3)
    assert a is not None and a.choice == 1


def test_unknown_version_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown prompt version"):
        build_prompt(rec("r", "x"), [rec("a", "y")], "v9")


# ---- selection ----


def result(rid: str, choice: int | str, conf: float) -> ArbiterResult:
    return ArbiterResult(
        rid, (f"{rid}a", f"{rid}b"), ArbiterAnswer(choice=choice, confidence=conf), None, 1, 1, 0.1
    )  # type: ignore[arg-type]


def val_and_decisions(truth: dict[str, set[str]]) -> tuple[ValSplit, dict[str, Decision]]:
    rows = tuple(Record(id=r, name=r) for r in truth)
    val = ValSplit(rows=rows, truth={r: frozenset(t) for r, t in truth.items()})
    return val, {r: Decision(r, Zone.LLM_ARBITER, None, "human") for r in truth}


def test_grid_cells_count_resolved_by_threshold() -> None:
    val, dec = val_and_decisions({"r1": {"r1b"}, "r2": {"r2a"}, "r3": set()})
    results = [result("r1", 2, 0.95), result("r2", 2, 0.55), result("r3", "none", 0.85)]
    cells = {c.min_confidence: c for c in grid_cells("v1", results, val, dec)}
    assert cells[0.5].resolved == 3 and cells[0.5].resolved_accuracy == pytest.approx(2 / 3)
    assert cells[0.8].resolved == 2 and cells[0.8].resolved_accuracy == 1.0
    assert cells[0.95].resolved == 1 and cells[0.95].match_precision == 1.0
    assert cells[0.8].none_precision == 1.0
    assert len(cells) == len(CONFIDENCE_GRID)
    # a human answers the unresolved rows, so system recall counts them as found
    assert cells[0.95].human_share == pytest.approx(2 / 3)


def test_select_prefers_the_widest_feasible_cell() -> None:
    val, dec = val_and_decisions({f"r{i}": {f"r{i}b"} for i in range(20)})
    good = [result(f"r{i}", 2, 0.9) for i in range(20)]  # all right, conf 0.9
    bad = [result(f"r{i}", 1, 0.9) for i in range(20)]  # all wrong
    cfg = LLMConfig(target_accuracy=0.9, min_resolved_support=5)
    res = tune({"v1": bad, "v2": good}, val, dec, cfg)
    assert res.feasible and res.chosen_version == "v2"
    assert res.chosen_min_confidence == 0.9  # highest threshold that still resolves all rows


def test_select_reports_infeasible_and_picks_the_most_accurate() -> None:
    val, dec = val_and_decisions({f"r{i}": {f"r{i}b"} for i in range(20)})
    half = [result(f"r{i}", 2 if i < 12 else 1, 0.9) for i in range(20)]  # 60% right
    cfg = LLMConfig(target_accuracy=0.95, min_resolved_support=5)
    res = tune({"v1": half}, val, dec, cfg)
    assert not res.feasible and res.chosen_version == "v1"


def test_select_ignores_cells_below_min_support() -> None:
    val, dec = val_and_decisions({"r1": {"r1b"}, "r2": {"r2b"}})
    results = [result("r1", 2, 0.9), result("r2", 2, 0.9)]
    cfg = LLMConfig(target_accuracy=0.5, min_resolved_support=10)
    res = tune({"v1": results}, val, dec, cfg)
    assert res.chosen_version is None and not res.feasible


def test_select_is_deterministic_on_ties() -> None:
    val, dec = val_and_decisions({f"r{i}": {f"r{i}b"} for i in range(12)})
    same = [result(f"r{i}", 2, 0.9) for i in range(12)]
    cfg = LLMConfig(target_accuracy=0.9, min_resolved_support=5)
    a = tune({"v3": same, "v2": same}, val, dec, cfg)
    b = tune({"v2": same, "v3": same}, val, dec, cfg)
    assert (a.chosen_version, a.chosen_min_confidence) == (
        b.chosen_version,
        b.chosen_min_confidence,
    )
    assert a.chosen_version == "v2"  # earlier version wins a tie


def test_select_function_on_empty_cells() -> None:
    assert select([], 0.9, 1) == (None, False)


# ---- the tuning code cannot reach test ----


@pytest.mark.parametrize("name", ["llm_tuning.py"])
def test_tuning_module_never_references_test(name: str) -> None:
    path: Path = PROJECT_ROOT / "src" / "entity_cascade" / "eval" / name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in {"TestSplit", "test"}:
            pytest.fail(f"{name} references {node.id!r}")
        if isinstance(node, ast.Attribute) and node.attr in {"test", "TestSplit"}:
            pytest.fail(f"{name} accesses .{node.attr}")


def test_tune_signature_takes_only_val_data() -> None:
    import inspect

    params = inspect.signature(tune).parameters
    assert list(params) == ["results_by_version", "val", "decisions_val", "cfg"]
    assert params["val"].annotation in (ValSplit, "ValSplit")
