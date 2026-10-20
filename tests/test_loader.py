from pathlib import Path

import pytest

from entity_cascade.config import PROJECT_ROOT, load_config
from entity_cascade.data.loader import load_dataset, load_mapping, read_csv_rows, read_text_auto


def test_read_text_auto_prefers_utf8(tmp_path: Path) -> None:
    p = tmp_path / "a.csv"
    p.write_bytes("id,name\n1,Sony®\n".encode())
    text, enc = read_text_auto(p)
    assert enc == "utf-8-sig"
    assert "Sony®" in text


def test_read_text_auto_falls_back_to_cp1252(tmp_path: Path) -> None:
    p = tmp_path / "a.csv"
    p.write_bytes(b'"id","name"\n"1","Sony\xae"\n')  # 0xAE is invalid utf-8
    text, enc = read_text_auto(p)
    assert enc == "cp1252"
    assert "Sony®" in text


def test_read_text_auto_falls_back_to_latin1_for_undefined_cp1252_bytes(tmp_path: Path) -> None:
    p = tmp_path / "a.csv"
    p.write_bytes(b'"id","name"\n"1","x\x81"\n')  # 0x81 is undefined in cp1252
    _, enc = read_text_auto(p)
    assert enc == "latin-1"


def test_read_csv_rows_handles_bom_quotes_and_multiline(tmp_path: Path) -> None:
    p = tmp_path / "a.csv"
    p.write_bytes('﻿"id","name"\n"1","a, b"\n"2","line1\nline2"\n'.encode())
    rows = read_csv_rows(p)
    assert rows == [{"id": "1", "name": "a, b"}, {"id": "2", "name": "line1\nline2"}]


def test_load_mapping_groups_multiple_matches(tmp_path: Path) -> None:
    p = tmp_path / "m.csv"
    p.write_text('"a","b"\n"1","x"\n"1","y"\n"2","z"\n', encoding="utf-8")
    assert load_mapping(p, "a", "b") == {"1": frozenset({"x", "y"}), "2": frozenset({"z"})}


def test_load_fixture_dataset() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "fixture.yaml")
    ds = load_dataset(cfg.dataset)
    assert len(ds.left) == 50
    assert len(ds.right) >= 50
    right_ids = {r.id for r in ds.right}
    assert all(t <= right_ids for t in ds.truth.values())
    assert all(ds.truth[r.id] for r in ds.left)  # every fixture row has a true pair
    # brand is derived for the left side from the catalog vocabulary
    assert sum(r.brand is not None for r in ds.left) >= 40


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    from entity_cascade.data.loader import load_records

    p = tmp_path / "d.csv"
    p.write_text('"id","name"\n"1","a"\n"1","b"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        load_records(p, "id", "name")
