import math

import pytest
from qdrant_client import QdrantClient

from entity_cascade.config import PROJECT_ROOT, load_config
from entity_cascade.data.loader import load_dataset
from entity_cascade.data.split import make_splits
from entity_cascade.eval.metrics import candidate_recall
from entity_cascade.retrieval.bm25 import BM25Index
from entity_cascade.retrieval.candidates import CandidateGenerator
from entity_cascade.retrieval.dense import DenseIndex
from entity_cascade.retrieval.rrf import rrf_fuse
from entity_cascade.retrieval.text import record_text
from entity_cascade.schemas import Record
from helpers import HashingEmbedder

DOCS = {
    "a": "Sony Turntable PSLX350H belt drive",
    "b": "Linksys EtherFast EZXS88W Ethernet Switch",
    "c": "Canon PowerShot SD1100 digital camera",
}


# ---- RRF ----


def test_rrf_scores_and_order() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    ids = [i for i, _ in fused]
    assert ids[:2] == ["a", "b"]  # equal scores; tie broken by best rank then id
    assert dict(fused)["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert dict(fused)["d"] == pytest.approx(1 / 63)


def test_rrf_item_in_both_lists_beats_item_in_one() -> None:
    fused = rrf_fuse([["x", "y"], ["z", "y"]], k=60)
    assert fused[0][0] == "y"


def test_rrf_top_and_empty() -> None:
    assert rrf_fuse([]) == []
    assert len(rrf_fuse([["a", "b", "c"]], top=2)) == 2


def test_rrf_is_deterministic() -> None:
    r = [["a", "b"], ["c", "d"]]
    assert rrf_fuse(r) == rrf_fuse(r)


# ---- BM25 ----


def test_bm25_finds_exact_model_code() -> None:
    idx = BM25Index(list(DOCS), list(DOCS.values()))
    assert idx.search("Sony PSLX350H turntable", 3)[0][0] == "a"
    assert idx.search("EZXS88W", 1)[0][0] == "b"


def test_bm25_matches_hyphenated_code_against_joined_code() -> None:
    idx = BM25Index(["x", "y"], ["Panasonic DMCFZ18K camera", "Panasonic microwave oven"])
    assert idx.search("DMC-FZ18K", 1)[0][0] == "x"


def test_bm25_empty_query_and_length_mismatch() -> None:
    idx = BM25Index(list(DOCS), list(DOCS.values()))
    assert idx.search("   ", 3) == []
    with pytest.raises(ValueError, match="differ in length"):
        BM25Index(["a"], ["x", "y"])


# ---- dense (Qdrant in memory, hashing embedder) ----


def test_dense_index_roundtrip() -> None:
    idx = DenseIndex(QdrantClient(":memory:"), "t", HashingEmbedder())
    idx.build(list(DOCS), list(DOCS.values()))
    hits = idx.search_many(["EZXS88W ethernet switch", "canon camera"], 2)
    assert hits[0][0][0] == "b"
    assert hits[1][0][0] == "c"
    assert len(hits[0]) == 2


def test_dense_index_rebuild_replaces_collection() -> None:
    idx = DenseIndex(QdrantClient(":memory:"), "t", HashingEmbedder())
    idx.build(["a"], ["alpha"])
    idx.build(["b"], ["beta"])
    assert [rid for rid, _ in idx.search_many(["beta"], 5)[0]] == ["b"]


# ---- candidate generator + recall ----


def test_candidate_generator_on_fixture_with_hashing_embedder() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "fixture.yaml")
    sp = make_splits(load_dataset(cfg.dataset), cfg.val_fraction, cfg.seed, 0.0)
    gen = CandidateGenerator(sp.catalog, cfg, HashingEmbedder(), QdrantClient(":memory:"))
    rows = sp.val.rows + sp.test.rows
    cands = gen.retrieve(rows)
    assert set(cands) == {r.id for r in rows}
    k = cfg.retrieval.top_k
    for c in cands.values():
        assert len(c.hybrid) == min(k, len(sp.catalog))
        assert len(set(c.hybrid)) == len(c.hybrid)
        assert c.hybrid_scores == tuple(sorted(c.hybrid_scores, reverse=True))
    truth = {**sp.val.truth, **sp.test.truth}
    rec = candidate_recall({i: c.hybrid for i, c in cands.items()}, truth)
    assert rec[1] <= rec[5] <= rec[20]
    assert rec[20] > 0.8


@pytest.mark.slow
def test_real_models_on_fixture() -> None:
    from entity_cascade.retrieval.dense import SentenceTransformerEmbedder

    cfg = load_config(PROJECT_ROOT / "configs" / "fixture.yaml")
    sp = make_splits(load_dataset(cfg.dataset), cfg.val_fraction, cfg.seed, 0.0)
    emb = SentenceTransformerEmbedder(cfg.models.embedding, cfg.models.query_instruction)
    gen = CandidateGenerator(sp.catalog, cfg, emb, QdrantClient(":memory:"))
    cands = gen.retrieve(sp.val.rows + sp.test.rows)
    truth = {**sp.val.truth, **sp.test.truth}
    hybrid = candidate_recall({i: c.hybrid for i, c in cands.items()}, truth)
    assert hybrid[20] >= 0.95
    assert hybrid[1] >= 0.6


# ---- metrics ----


def test_candidate_recall_definition() -> None:
    truth = {"r1": frozenset({"a"}), "r2": frozenset({"b", "c"}), "r3": frozenset()}
    rankings = {"r1": ["x", "a"], "r2": ["x", "y", "c"], "r3": ["a"]}
    rec = candidate_recall(rankings, truth, ks=(1, 2, 3))
    # r3 has no true pair and is excluded: denominator is 2
    assert rec == {1: 0.0, 2: 0.5, 3: 1.0}


def test_candidate_recall_no_rows_is_nan() -> None:
    assert math.isnan(candidate_recall({}, {"r": frozenset()}, ks=(1,))[1])


def test_record_text_falls_back_to_name() -> None:
    rec = Record(id="1", name="Only Name", fields={"description": ""})
    assert record_text(rec, ["description"]) == "Only Name"
    rec2 = Record(id="2", name="n", fields={"name": "n", "description": "d"})
    assert record_text(rec2, ["name", "description"]) == "n d"
