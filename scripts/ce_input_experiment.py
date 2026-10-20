"""Which text should the cross-encoder see? Val-only experiment behind rerank.query_fields.

Prints top-1 accuracy on the val split only (rows with a pair) for three query/document
variants. The test split is never scored here. Takes a few minutes on CPU (no score cache).
"""

from __future__ import annotations

import argparse
import time

from entity_cascade.config import PROJECT_ROOT, load_config
from entity_cascade.data.loader import load_dataset
from entity_cascade.data.split import make_splits
from entity_cascade.eval.metrics import top1_accuracy
from entity_cascade.rerank.cross_encoder import CrossEncoderReranker
from entity_cascade.retrieval.candidates import CandidateGenerator
from entity_cascade.retrieval.dense import SentenceTransformerEmbedder, make_client
from entity_cascade.retrieval.text import record_text

VARIANTS: dict[str, tuple[list[str], list[str]]] = {
    "query=name, doc=name+manufacturer": (["name"], ["name", "manufacturer"]),
    "query=name, doc=name+description+manufacturer": (
        ["name"],
        ["name", "description", "manufacturer"],
    ),
    "query=name+description, doc=name+description+manufacturer": (
        ["name", "description"],
        ["name", "description", "manufacturer"],
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    sp = make_splits(
        load_dataset(cfg.dataset), cfg.val_fraction, cfg.seed, cfg.dataset.new_product_fraction
    )
    embedder = SentenceTransformerEmbedder(cfg.models.embedding, cfg.models.query_instruction)
    gen = CandidateGenerator(sp.catalog, cfg, embedder, make_client(":memory:"))
    rows = sp.val.rows
    cands = gen.retrieve(rows)
    catalog = {r.id: r for r in sp.catalog}
    reranker = CrossEncoderReranker(cfg.models.reranker, cfg.rerank.batch_size)

    for label, (query_fields, doc_fields) in VARIANTS.items():
        t0 = time.perf_counter()
        pairs: list[tuple[str, str]] = []
        index: list[tuple[str, str]] = []
        for row in rows:
            for cid in cands[row.id].hybrid:
                pairs.append(
                    (record_text(row, query_fields), record_text(catalog[cid], doc_fields))
                )
                index.append((row.id, cid))
        best: dict[str, tuple[float, str]] = {}
        for (rid, cid), score in zip(index, reranker.score(pairs), strict=True):
            if rid not in best or score > best[rid][0]:
                best[rid] = (score, cid)
        acc, n = top1_accuracy({k: v[1] for k, v in best.items()}, dict(sp.val.truth))
        print(
            f"{label}: val top-1 {acc:.1%} (n={n}) in {time.perf_counter() - t0:.0f}s", flush=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
