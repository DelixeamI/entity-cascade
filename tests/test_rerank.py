from collections.abc import Sequence
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from entity_cascade.config import PROJECT_ROOT, load_config
from entity_cascade.data.loader import load_dataset
from entity_cascade.data.split import ValSplit, make_splits
from entity_cascade.pipeline import compute_ranking_inputs
from entity_cascade.rerank.cross_encoder import CachedReranker
from entity_cascade.rerank.features import (
    FeatureWeights,
    MatchFeatures,
    compute_features,
)
from entity_cascade.rerank.ranker import RawCandidate, RawRow, rank_row
from entity_cascade.rerank.weights import fit_feature_weights
from entity_cascade.retrieval.candidates import CandidateGenerator
from entity_cascade.schemas import Record
from helpers import HashingEmbedder


def rec(rid: str, name: str, brand: str | None = None, desc: str = "") -> Record:
    return Record(id=rid, name=name, description=desc, brand=brand)


NO_FEATURES = MatchFeatures(
    code_exact=False,
    code_prefix=False,
    code_conflict=False,
    brand_match=False,
    brand_conflict=False,
)


# ---- features ----


def test_code_exact_and_brand_match() -> None:
    f = compute_features(
        rec("1", "Sony Turntable - PSLX350H", brand="sony"),
        rec("2", "Sony PSLX350H Belt Drive Turntable", brand="SONY"),
    )
    assert f.code_exact and not f.code_prefix and not f.code_conflict
    assert f.brand_match and not f.brand_conflict


def test_code_prefix_when_one_side_has_variant_suffix() -> None:
    f = compute_features(
        rec("1", "Sony Vertical-In-The-Ear Headphones - MDRJ10"),
        rec("2", "Sony h.ear Earphones - MDRJ10/BLUE"),
    )
    assert f.code_prefix and not f.code_exact and not f.code_conflict


def test_short_codes_do_not_count_as_prefix() -> None:
    f = compute_features(rec("1", "Thing AB12"), rec("2", "Thing AB123X"))
    assert not f.code_prefix  # "ab12" is shorter than the minimum prefix length


def test_code_conflict_and_brand_conflict() -> None:
    f = compute_features(
        rec("1", "Canon camera SD1100", brand="canon"),
        rec("2", "Nikon camera S210", brand="nikon"),
    )
    assert f.code_conflict and f.brand_conflict
    assert not f.code_exact and not f.brand_match


def test_missing_codes_and_brands_are_neutral() -> None:
    f = compute_features(rec("1", "Plain headphones"), rec("2", "Other headphones"))
    assert f == NO_FEATURES


def test_weights_adjust_signs() -> None:
    w = FeatureWeights(code_exact=3, code_conflict=2, brand_conflict=1, brand_match=1)
    bad = MatchFeatures(
        code_exact=False,
        code_prefix=False,
        code_conflict=True,
        brand_match=False,
        brand_conflict=True,
    )
    good = MatchFeatures(
        code_exact=True,
        code_prefix=False,
        code_conflict=False,
        brand_match=True,
        brand_conflict=False,
    )
    assert w.adjust(bad) == -3.0
    assert w.adjust(good) == 4.0


# ---- ranking, margin ----


def raw(rid: str, items: list[tuple[str, float, MatchFeatures]]) -> RawRow:
    return RawRow(rid, tuple(RawCandidate(c, i + 1, s, f) for i, (c, s, f) in enumerate(items)))


def test_rank_row_orders_by_final_score_and_computes_margin() -> None:
    row = raw("r", [("a", 1.0, NO_FEATURES), ("b", 0.5, NO_FEATURES), ("c", -2.0, NO_FEATURES)])
    ranked = rank_row(row, FeatureWeights())
    assert [c.catalog_id for c in ranked.candidates] == ["a", "b", "c"]
    assert ranked.top1_id == "a"
    assert ranked.score == 1.0
    assert ranked.margin == pytest.approx(0.5)


def test_features_can_overturn_the_cross_encoder() -> None:
    exact = NO_FEATURES.model_copy(update={"code_exact": True})
    row = raw("r", [("a", 1.0, NO_FEATURES), ("b", 0.0, exact)])
    assert rank_row(row, FeatureWeights()).top1_id == "a"
    assert rank_row(row, FeatureWeights(code_exact=2.0)).top1_id == "b"


def test_ties_keep_rrf_order_and_single_candidate_margin() -> None:
    row = raw("r", [("a", 1.0, NO_FEATURES), ("b", 1.0, NO_FEATURES)])
    ranked = rank_row(row, FeatureWeights())
    assert ranked.top1_id == "a"
    assert ranked.margin == 0.0
    single = rank_row(raw("r", [("a", 2.5, NO_FEATURES)]), FeatureWeights())
    assert single.margin == 2.5
    empty = rank_row(RawRow("r", ()), FeatureWeights())
    assert empty.top1_id is None
    assert empty.margin == 0.0


# ---- weight fitting (val only) ----


def _val(rows: dict[str, set[str]]) -> ValSplit:
    return ValSplit(
        rows=tuple(rec(r, f"row {r}") for r in rows),
        truth={r: frozenset(t) for r, t in rows.items()},
    )


def test_fit_weights_learns_that_exact_code_beats_a_noisy_cross_encoder() -> None:
    exact = NO_FEATURES.model_copy(update={"code_exact": True})
    # The cross-encoder prefers the wrong candidate; the exact code marks the right one.
    raws = {
        f"r{i}": raw(f"r{i}", [("wrong", 1.0, NO_FEATURES), ("right", 0.0, exact)])
        for i in range(10)
    }
    val = _val({f"r{i}": {"right"} for i in range(10)})
    w = fit_feature_weights(val, raws, grid=(0.0, 1.0, 2.0))
    assert w.code_exact > 1.0
    final = {c.catalog_id: c.final_score for c in rank_row(raws["r0"], w).candidates}
    assert final["right"] > final["wrong"]


def test_fit_weights_prefers_smaller_weights_on_ties() -> None:
    raws = {"r": raw("r", [("a", 1.0, NO_FEATURES), ("b", 0.0, NO_FEATURES)])}
    w = fit_feature_weights(_val({"r": {"a"}}), raws, grid=(0.0, 1.0, 2.0))
    assert w == FeatureWeights()


def test_fit_weights_ignores_rows_without_pair() -> None:
    raws = {"r": raw("r", [("a", 1.0, NO_FEATURES)])}
    assert fit_feature_weights(_val({"r": set()}), raws, grid=(0.0, 1.0)) == FeatureWeights()


# ---- cached reranker ----


class CountingReranker:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.calls.append(len(pairs))
        return [float(len(q) + 2 * len(d)) for q, d in pairs]


def test_cached_reranker_returns_identical_scores_and_skips_recompute(tmp_path: Path) -> None:
    inner = CountingReranker()
    path = tmp_path / "c.jsonl"
    cached = CachedReranker(inner, "m", path)
    pairs = [("a", "bb"), ("ccc", "d")]
    first = cached.score(pairs)
    second = cached.score([*pairs, ("x", "y")])
    assert second[:2] == first
    assert inner.calls == [2, 1]
    assert (cached.hits, cached.misses) == (2, 3)

    reloaded = CachedReranker(CountingReranker(), "m", path)  # persisted across instances
    assert reloaded.score(pairs) == first
    assert reloaded.misses == 0


def test_cached_reranker_key_depends_on_model_name(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    CachedReranker(CountingReranker(), "m1", path).score([("a", "b")])
    other = CachedReranker(CountingReranker(), "m2", path)
    other.score([("a", "b")])
    assert other.misses == 1


# ---- end to end on the fixture with fake models ----


class LengthReranker:
    """Deterministic fake: more shared tokens means a higher score."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [float(len(set(q.lower().split()) & set(d.lower().split()))) for q, d in pairs]


def test_compute_ranking_inputs_on_fixture() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "fixture.yaml")
    sp = make_splits(load_dataset(cfg.dataset), cfg.val_fraction, cfg.seed, 0.0)
    gen = CandidateGenerator(sp.catalog, cfg, HashingEmbedder(), QdrantClient(":memory:"))
    out = compute_ranking_inputs(cfg, gen, sp.catalog, sp.val.rows, LengthReranker())
    assert set(out.raws) == {r.id for r in sp.val.rows}
    for row in sp.val.rows:
        items = out.raws[row.id].items
        assert [c.catalog_id for c in items] == list(out.cands[row.id].hybrid)
        assert [c.hybrid_rank for c in items] == list(range(1, len(items) + 1))
    assert out.rerank_ms_per_row >= 0.0
