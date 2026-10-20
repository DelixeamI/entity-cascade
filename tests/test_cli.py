from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from entity_cascade.cli import (
    OUTPUT_COLUMNS,
    cmd_match,
    main,
    read_price_list,
    summarize,
    write_result,
)
from entity_cascade.data.loader import load_dataset
from entity_cascade.eval.run import Models, run_cascade
from entity_cascade.llm.fake import HeuristicFakeLLM
from entity_cascade.matcher import match_rows
from entity_cascade.retrieval.candidates import CandidateGenerator
from entity_cascade.schemas import Zone
from helpers import HashingEmbedder
from test_run import OverlapReranker, fixture_cfg


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    cfg = fixture_cfg(tmp_path_factory.mktemp("cli"))
    run = run_cascade(cfg, Models(HashingEmbedder(), OverlapReranker(), OverlapReranker()))
    catalog = list(load_dataset(cfg.dataset).right)
    gen = CandidateGenerator(catalog, cfg, HashingEmbedder(), QdrantClient(":memory:"))
    return cfg, run, catalog, gen


def write_price_list(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_match_rows_end_to_end(setup, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    cfg, run, catalog, gen = setup
    src = [
        {"id": f"row{i}", "name": r.name, "description": r.description, "price": "10"}
        for i, r in enumerate(run.splits.test.rows[:12])
    ]
    inp = tmp_path / "price.csv"
    write_price_list(inp, src)
    rows = read_price_list(inp, cfg, "id", "name", "description", catalog)
    outcomes, note = match_rows(
        cfg, run.cal, rows, catalog, gen, OverlapReranker(), HeuristicFakeLLM()
    )
    assert note is None
    assert [o.row.id for o in outcomes] == [r.id for r in rows]  # input order kept
    for o in outcomes:
        assert o.reason
        assert (o.verdict == "match") == (o.candidate is not None and o.verdict == "match")
        if o.zone in (Zone.AUTO_MATCH, Zone.NO_MATCH):
            assert o.handled_by == "auto"
        if o.handled_by == "human":
            assert o.verdict == "needs_human"

    out = tmp_path / "result.csv"
    write_result(out, outcomes)
    with out.open(encoding="utf-8", newline="") as f:
        written = list(csv.DictReader(f))
    assert list(written[0]) == OUTPUT_COLUMNS
    assert len(written) == 12
    for w in written:
        if w["verdict"] != "match":
            assert w["matched_id"] == ""

    text = summarize(outcomes, cfg, 1.5, None, None)
    assert "rows: 12" in text and "zones:" in text and "handled:" in text


def test_no_llm_sends_gray_rows_to_a_human(setup) -> None:  # type: ignore[no-untyped-def]
    cfg, run, catalog, gen = setup
    outcomes, _ = match_rows(
        cfg, run.cal, list(run.splits.test.rows), catalog, gen, OverlapReranker(), None
    )
    gray = [o for o in outcomes if o.zone is Zone.LLM_ARBITER]
    assert gray
    assert all(o.handled_by == "human" and "no LLM was used" in o.reason for o in gray)


def test_missing_cassettes_degrade_instead_of_crashing(setup) -> None:  # type: ignore[no-untyped-def]
    from entity_cascade.llm.recorded import RecordedLLM

    cfg, run, catalog, gen = setup
    llm = RecordedLLM("nothing-recorded", cfg.llm.cassette_dir / "empty")
    outcomes, note = match_rows(
        cfg, run.cal, list(run.splits.test.rows), catalog, gen, OverlapReranker(), llm
    )
    assert note is not None and "make record" in note
    assert all(o.handled_by != "llm" for o in outcomes)


def test_read_price_list_validation(setup, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    cfg, _, catalog, _ = setup
    p = tmp_path / "p.csv"
    write_price_list(p, [{"id": "1", "title": "x"}])
    with pytest.raises(ValueError, match="no column 'name'"):
        read_price_list(p, cfg, "id", "name", "description", catalog)

    write_price_list(p, [{"id": "1", "name": "a"}, {"id": "1", "name": "b"}])
    with pytest.raises(ValueError, match="duplicate row id"):
        read_price_list(p, cfg, "id", "name", "description", catalog)

    p.write_text('"id","name"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        read_price_list(p, cfg, "id", "name", "description", catalog)


def test_ids_default_to_row_numbers(setup, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    cfg, _, catalog, _ = setup
    p = tmp_path / "p.csv"
    write_price_list(p, [{"name": "a"}, {"name": "b"}])
    rows = read_price_list(p, cfg, "id", "name", "description", catalog)
    assert [r.id for r in rows] == ["1", "2"]


def test_cmd_match_without_calibration_explains_what_to_do(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    args = argparse.Namespace(
        config="configs/fixture.yaml",
        calibration=str(tmp_path / "missing.json"),
        demo=0,
        input=None,
        output=str(tmp_path / "o.csv"),
    )
    assert cmd_match(args) == 1
    assert "make eval" in capsys.readouterr().out


def test_main_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([])
