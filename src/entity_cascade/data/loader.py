"""Read benchmark CSV files (utf-8 or legacy 8-bit encodings) into typed records."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from entity_cascade.config import DatasetConfig, resolve_path
from entity_cascade.data.normalize import build_brand_vocab, extract_brand
from entity_cascade.schemas import Record

_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def read_text_auto(path: Path) -> tuple[str, str]:
    """Decode a file trying strict utf-8, then cp1252, then latin-1. Returns (text, encoding).

    Abt.csv is not valid utf-8 (byte 0xAE is the registered sign in cp1252/latin-1), so a blind
    utf-8 read would crash and a blind latin-1 read would garble the other files.
    """
    raw = path.read_bytes()
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("latin-1", raw, 0, 1, f"cannot decode {path}")  # pragma: no cover


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    text, _ = read_text_auto(path)
    reader = csv.DictReader(io.StringIO(text, newline=""))
    return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]


def load_records(
    path: Path,
    id_col: str,
    name_col: str,
    brand_col: str | None = None,
) -> list[Record]:
    records: list[Record] = []
    seen: set[str] = set()
    for row in read_csv_rows(path):
        rid = row[id_col]
        if rid in seen:
            raise ValueError(f"duplicate id {rid!r} in {path.name}")
        seen.add(rid)
        brand = row.get(brand_col) or None if brand_col else None
        records.append(
            Record(
                id=rid,
                name=row.get(name_col, ""),
                description=row.get("description", ""),
                brand=brand,
                fields=row,
            )
        )
    return records


def load_mapping(path: Path, left_col: str, right_col: str) -> dict[str, frozenset[str]]:
    acc: dict[str, set[str]] = {}
    for row in read_csv_rows(path):
        acc.setdefault(row[left_col], set()).add(row[right_col])
    return {k: frozenset(v) for k, v in acc.items()}


@dataclass(frozen=True)
class Dataset:
    left: tuple[Record, ...]  # price-list rows (Abt)
    right: tuple[Record, ...]  # catalog (Buy)
    truth: dict[str, frozenset[str]]  # left id -> true catalog ids (empty = no pair)


def load_dataset(cfg: DatasetConfig) -> Dataset:
    base = resolve_path(cfg.data_dir)
    right = load_records(base / cfg.right_file, cfg.right_id, cfg.right_name, cfg.right_brand)
    left = load_records(base / cfg.left_file, cfg.left_id, cfg.left_name, cfg.left_brand)
    mapping = load_mapping(base / cfg.mapping_file, cfg.mapping_left_col, cfg.mapping_right_col)

    left_ids = {r.id for r in left}
    right_ids = {r.id for r in right}
    for lid, rids in mapping.items():
        if lid not in left_ids:
            raise ValueError(f"mapping references unknown left id {lid!r}")
        unknown = rids - right_ids
        if unknown:
            raise ValueError(f"mapping references unknown right ids {sorted(unknown)[:3]}")

    # The left side often has no brand (or an empty one): derive it from the catalog's brand
    # vocabulary, only where the data does not already give one.
    if cfg.right_brand is not None:
        vocab = build_brand_vocab(r.brand for r in right)
        left = [
            r if r.brand else r.model_copy(update={"brand": extract_brand(r.name, vocab)})
            for r in left
        ]

    truth = {r.id: mapping.get(r.id, frozenset()) for r in left}
    return Dataset(left=tuple(left), right=tuple(right), truth=truth)
