"""Thresholds and weights are fitted on val only; test labels must not influence them."""

from __future__ import annotations

import ast
import random
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from entity_cascade.calibration import Calibration
from entity_cascade.config import PROJECT_ROOT, AppConfig, load_config
from entity_cascade.data.loader import load_dataset
from entity_cascade.data.split import Splits, TestSplit, ValSplit, make_splits
from entity_cascade.pipeline import fit_on_val
from entity_cascade.rerank.features import MatchFeatures
from entity_cascade.rerank.ranker import RawCandidate, RawRow

SRC = PROJECT_ROOT / "src" / "entity_cascade"
FITTING_MODULES = [
    SRC / "calibration.py",
    SRC / "zones" / "thresholds.py",
    SRC / "zones" / "assign.py",
    SRC / "rerank" / "weights.py",
]
NF = MatchFeatures(
    code_exact=False,
    code_prefix=False,
    code_conflict=False,
    brand_match=False,
    brand_conflict=False,
)


class ForbiddenTruth(Mapping[str, frozenset[str]]):
    """A truth mapping that fails the test on any access."""

    def __getitem__(self, key: str) -> frozenset[str]:
        raise AssertionError(f"test label read during calibration: {key!r}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("test labels iterated during calibration")

    def __len__(self) -> int:
        raise AssertionError("test labels counted during calibration")

    def get(self, key: Any, default: Any = None) -> Any:
        raise AssertionError("test labels read during calibration")


def synthetic_raws(split: ValSplit | TestSplit, seed: int) -> dict[str, RawRow]:
    """Cross-encoder scores correlated with truth: true item high, distractors low."""
    rng = random.Random(seed)
    out: dict[str, RawRow] = {}
    for row in split.rows:
        truth = split.truth[row.id] if not isinstance(split.truth, ForbiddenTruth) else frozenset()
        items: list[RawCandidate] = []
        for t in sorted(truth):
            items.append(RawCandidate(t, 0, rng.uniform(2, 12), NF))
        for i in range(5):
            top = 4.0 if truth else 0.0
            items.append(RawCandidate(f"{row.id}_d{i}", 0, rng.uniform(-8, top), NF))
        rng.shuffle(items)
        out[row.id] = RawRow(
            row.id,
            tuple(RawCandidate(c.catalog_id, i + 1, c.ce_score, NF) for i, c in enumerate(items)),
        )
    return out


def setup() -> tuple[AppConfig, Splits, dict[str, RawRow], dict[str, RawRow]]:
    cfg = load_config(PROJECT_ROOT / "configs" / "fixture.yaml")
    sp = make_splits(load_dataset(cfg.dataset), cfg.val_fraction, cfg.seed, 0.2)
    return cfg, sp, synthetic_raws(sp.val, 1), synthetic_raws(sp.test, 2)


def with_test_truth(sp: Splits, truth: Mapping[str, frozenset[str]]) -> Splits:
    return Splits(
        catalog=sp.catalog,
        val=sp.val,
        test=TestSplit(rows=sp.test.rows, truth=truth),
        new_product_ids=sp.new_product_ids,
    )


def test_fit_on_val_never_reads_test_labels() -> None:
    cfg, sp, val_raws, _ = setup()
    booby_trapped = with_test_truth(sp, ForbiddenTruth())
    cal = fit_on_val(booby_trapped, val_raws, cfg)  # would raise on any test-label access
    assert cal.thresholds.auto_feasible  # and the fit is not vacuous


def test_calibration_is_identical_when_test_labels_are_shuffled() -> None:
    cfg, sp, val_raws, _ = setup()
    baseline = fit_on_val(sp, val_raws, cfg)

    ids = list(sp.test.truth)
    permuted = ids[:]
    random.Random(99).shuffle(permuted)
    shuffled = {i: sp.test.truth[j] for i, j in zip(ids, permuted, strict=True)}
    assert shuffled != dict(sp.test.truth)  # the shuffle really changed the labels
    assert fit_on_val(with_test_truth(sp, shuffled), val_raws, cfg) == baseline

    emptied = {i: frozenset[str]() for i in ids}
    assert fit_on_val(with_test_truth(sp, emptied), val_raws, cfg) == baseline


def test_calibration_does_react_to_val_labels() -> None:
    """Guards the tests above against passing vacuously: val labels do matter."""
    cfg, sp, val_raws, _ = setup()
    baseline = fit_on_val(sp, val_raws, cfg)
    ids = list(sp.val.truth)
    rotated = {ids[i]: sp.val.truth[ids[(i + 7) % len(ids)]] for i in range(len(ids))}
    other = Splits(
        catalog=sp.catalog,
        val=ValSplit(rows=sp.val.rows, truth=rotated),
        test=sp.test,
        new_product_ids=sp.new_product_ids,
    )
    assert fit_on_val(other, val_raws, cfg) != baseline


def test_calibration_roundtrips_through_json(tmp_path: Path) -> None:
    cfg, sp, val_raws, _ = setup()
    cal = fit_on_val(sp, val_raws, cfg)
    cal.save(tmp_path / "c.json")
    assert Calibration.load(tmp_path / "c.json") == cal


def test_val_and_test_are_unrelated_types() -> None:
    assert not issubclass(TestSplit, ValSplit)
    assert not issubclass(ValSplit, TestSplit)


def test_val_and_test_rows_are_disjoint() -> None:
    _, sp, _, _ = setup()
    assert {r.id for r in sp.val.rows}.isdisjoint({r.id for r in sp.test.rows})


@pytest.mark.parametrize("path", FITTING_MODULES, ids=lambda p: p.name)
def test_fitting_modules_do_not_reference_the_test_split(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    banned = {"TestSplit", "test"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in banned:
            pytest.fail(f"{path.name} references {node.id!r} (line {node.lineno})")
        if isinstance(node, ast.Attribute) and node.attr in banned:
            pytest.fail(f"{path.name} accesses .{node.attr} (line {node.lineno})")
        if isinstance(node, ast.ImportFrom | ast.Import):
            for alias in node.names:
                if alias.name in banned:
                    pytest.fail(f"{path.name} imports {alias.name!r}")
