from entity_cascade.config import PROJECT_ROOT, load_config
from entity_cascade.data.loader import load_dataset
from entity_cascade.data.stats import dataset_stats


def test_fixture_stats_are_consistent() -> None:
    ds = load_dataset(load_config(PROJECT_ROOT / "configs" / "fixture.yaml").dataset)
    s = dataset_stats(ds)
    assert s.left_rows == 50
    assert s.left_with_pair + s.left_without_pair == s.left_rows
    assert s.pairs == sum(len(t) for t in ds.truth.values())
    assert s.pairs_sharing_code <= s.pairs
    assert s.left_with_multiple_pairs <= s.left_with_pair
