"""Descriptive statistics of a loaded dataset (reported, never hand-typed)."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from entity_cascade.data.loader import Dataset
from entity_cascade.data.normalize import extract_codes


class DatasetStats(BaseModel):
    left_rows: int
    right_rows: int
    pairs: int
    left_with_pair: int
    left_without_pair: int
    left_with_multiple_pairs: int
    right_with_multiple_left: int
    left_with_brand: int
    left_with_code: int
    pairs_sharing_code: int  # (left, right) true pairs with at least one equal model code


def _codes(name: str, description: str) -> set[str]:
    return set(extract_codes(name)) | set(extract_codes(description))


def dataset_stats(ds: Dataset) -> DatasetStats:
    right = {r.id: r for r in ds.right}
    right_partners = Counter(rid for t in ds.truth.values() for rid in t)
    left_codes = {r.id: _codes(r.name, r.description) for r in ds.left}
    pairs = sum(len(t) for t in ds.truth.values())
    sharing = sum(
        1
        for r in ds.left
        for rid in ds.truth[r.id]
        if left_codes[r.id] & _codes(right[rid].name, right[rid].description)
    )
    with_pair = sum(1 for t in ds.truth.values() if t)
    return DatasetStats(
        left_rows=len(ds.left),
        right_rows=len(ds.right),
        pairs=pairs,
        left_with_pair=with_pair,
        left_without_pair=len(ds.left) - with_pair,
        left_with_multiple_pairs=sum(1 for t in ds.truth.values() if len(t) > 1),
        right_with_multiple_left=sum(1 for c in right_partners.values() if c > 1),
        left_with_brand=sum(1 for r in ds.left if r.brand),
        left_with_code=sum(1 for c in left_codes.values() if c),
        pairs_sharing_code=sharing,
    )
