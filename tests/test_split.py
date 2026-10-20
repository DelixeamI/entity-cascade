import random

from entity_cascade.config import PROJECT_ROOT, load_config
from entity_cascade.data.loader import Dataset, load_dataset
from entity_cascade.data.split import make_splits, split_ids


def _fixture_ds() -> Dataset:
    return load_dataset(load_config(PROJECT_ROOT / "configs" / "fixture.yaml").dataset)


def test_split_is_disjoint_complete_and_sized() -> None:
    ids = [str(i) for i in range(101)]
    val, test = split_ids(ids, 0.5, seed=1)
    assert val.isdisjoint(test)
    assert val | test == set(ids)
    assert len(val) == round(101 * 0.5)


def test_split_is_deterministic_and_independent_of_input_order() -> None:
    ids = [str(i) for i in range(200)]
    shuffled = ids[:]
    random.Random(5).shuffle(shuffled)
    assert split_ids(ids, 0.5, seed=3) == split_ids(shuffled, 0.5, seed=3)
    assert split_ids(ids, 0.5, seed=3) != split_ids(ids, 0.5, seed=4)


def test_make_splits_without_new_products_keeps_catalog() -> None:
    ds = _fixture_ds()
    sp = make_splits(ds, 0.5, seed=13, new_product_fraction=0.0)
    assert sp.catalog == ds.right
    assert not sp.new_product_ids
    assert len(sp.val.rows) + len(sp.test.rows) == len(ds.left)
    assert {r.id for r in sp.val.rows}.isdisjoint({r.id for r in sp.test.rows})


def test_new_products_removed_from_catalog_and_truth_emptied() -> None:
    ds = _fixture_ds()
    sp = make_splits(ds, 0.5, seed=13, new_product_fraction=0.2)
    catalog_ids = {r.id for r in sp.catalog}
    assert sp.new_product_ids
    for lid in sp.new_product_ids:
        assert not ds.truth[lid] & catalog_ids  # true items are gone from the catalog
    for split in (sp.val, sp.test):
        for row in split.rows:
            if row.id in sp.new_product_ids:
                assert split.truth[row.id] == frozenset()
            else:
                # no other row loses its true items
                assert split.truth[row.id] == ds.truth[row.id]
                assert split.truth[row.id] <= catalog_ids


def test_new_products_share_per_split() -> None:
    ds = _fixture_ds()
    sp = make_splits(ds, 0.5, seed=13, new_product_fraction=0.2)
    val_new = sum(r.id in sp.new_product_ids for r in sp.val.rows)
    test_new = sum(r.id in sp.new_product_ids for r in sp.test.rows)
    assert val_new == round(0.2 * len(sp.val.rows))
    assert test_new == round(0.2 * len(sp.test.rows))
