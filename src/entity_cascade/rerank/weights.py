"""Choose feature weights by grid search on val top-1 accuracy."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np

from entity_cascade.data.split import ValSplit
from entity_cascade.rerank.features import FeatureWeights, feature_vector
from entity_cascade.rerank.ranker import RawRow

_NAMES = ("code_exact", "code_prefix", "code_conflict", "brand_match", "brand_conflict")


def fit_feature_weights(
    val: ValSplit, raws: Mapping[str, RawRow], grid: Sequence[float]
) -> FeatureWeights:
    """Pick the weight combination with the best val top-1 accuracy.

    Only rows that have a true pair count. Ties go to the smaller total weight (the simpler
    model), then to the grid order, so the result is deterministic. Takes a ValSplit by type:
    test rows cannot be passed in.
    """
    rows = [r for r in val.rows if val.truth[r.id] and raws[r.id].items]
    if not rows:
        return FeatureWeights()
    width = max(len(raws[r.id].items) for r in rows)
    n, n_feat = len(rows), len(_NAMES)
    ce = np.full((n, width), -np.inf)
    feats = np.zeros((n, width, n_feat))
    correct = np.zeros((n, width), dtype=bool)
    for i, r in enumerate(rows):
        for j, c in enumerate(raws[r.id].items):
            ce[i, j] = c.ce_score
            feats[i, j] = feature_vector(c.features)
            correct[i, j] = c.catalog_id in val.truth[r.id]

    best: tuple[float, float, tuple[float, ...]] | None = None
    for combo in itertools.product(grid, repeat=n_feat):
        final = ce + feats @ np.asarray(combo)
        top = np.argmax(final, axis=1)  # first max wins: RRF order breaks ties
        acc = float(correct[np.arange(n), top].mean())
        key = (-acc, sum(combo))
        if best is None or key < (best[0], best[1]):
            best = (key[0], key[1], combo)
    assert best is not None
    return FeatureWeights(**dict(zip(_NAMES, best[2], strict=True)))
