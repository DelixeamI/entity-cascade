"""Deterministic val/test split of price-list rows, plus synthetic "new product" rows."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass

from entity_cascade.data.loader import Dataset
from entity_cascade.schemas import Record


@dataclass(frozen=True)
class _SplitBase:
    rows: tuple[Record, ...]
    truth: Mapping[str, frozenset[str]]  # row id -> true catalog ids; empty = no pair


@dataclass(frozen=True)
class ValSplit(_SplitBase):
    """Rows used to choose thresholds. The only split threshold fitting may touch."""


@dataclass(frozen=True)
class TestSplit(_SplitBase):
    """Rows used only for final metrics. Never passed to threshold fitting."""

    __test__ = False  # not a pytest class


@dataclass(frozen=True)
class Splits:
    catalog: tuple[Record, ...]
    val: ValSplit
    test: TestSplit
    new_product_ids: frozenset[str]  # rows whose true items were removed from the catalog


def split_ids(ids: list[str], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    """Shuffle sorted ids with a seeded RNG and cut. Independent of input order."""
    ordered = sorted(ids)
    random.Random(seed).shuffle(ordered)
    n_val = round(len(ordered) * val_fraction)
    return set(ordered[:n_val]), set(ordered[n_val:])


def _pick_new_products(
    row_ids: list[str],
    truth: Mapping[str, frozenset[str]],
    partners: Mapping[str, set[str]],
    fraction: float,
    seed: int,
) -> set[str]:
    """Pick rows to turn into "new products". Only rows whose true items belong to no other
    row qualify, so removing them from the catalog cannot break another row's truth."""
    eligible = sorted(
        rid for rid in row_ids if truth[rid] and all(partners[c] == {rid} for c in truth[rid])
    )
    k = min(len(eligible), round(fraction * len(row_ids)))
    return set(random.Random(seed).sample(eligible, k))


def make_splits(
    ds: Dataset, val_fraction: float, seed: int, new_product_fraction: float = 0.0
) -> Splits:
    val_ids, test_ids = split_ids([r.id for r in ds.left], val_fraction, seed)

    partners: dict[str, set[str]] = {}
    for lid, rids in ds.truth.items():
        for rid in rids:
            partners.setdefault(rid, set()).add(lid)

    new_ids: set[str] = set()
    if new_product_fraction > 0.0:
        # Per split, so val and test both contain the same share of new products.
        for offset, ids in enumerate((val_ids, test_ids)):
            new_ids |= _pick_new_products(
                sorted(ids), ds.truth, partners, new_product_fraction, seed + 1 + offset
            )

    removed: set[str] = set()
    for lid in new_ids:
        removed |= ds.truth[lid]

    truth = {lid: (frozenset() if lid in new_ids else t) for lid, t in ds.truth.items()}
    catalog = tuple(r for r in ds.right if r.id not in removed)
    val_rows = tuple(r for r in ds.left if r.id in val_ids)
    test_rows = tuple(r for r in ds.left if r.id in test_ids)
    return Splits(
        catalog=catalog,
        val=ValSplit(rows=val_rows, truth={r.id: truth[r.id] for r in val_rows}),
        test=TestSplit(rows=test_rows, truth={r.id: truth[r.id] for r in test_rows}),
        new_product_ids=frozenset(new_ids),
    )
