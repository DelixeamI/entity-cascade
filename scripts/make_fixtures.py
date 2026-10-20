"""Build the small test fixture: N price-list rows, their true catalog items and hard distractors.

Distractors are the catalog items with the highest token overlap with the row's name, excluding
true matches. Output uses the same CSV layout as the raw data, so the normal loader reads it.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from entity_cascade.config import PROJECT_ROOT, load_config
from entity_cascade.data.loader import load_dataset
from entity_cascade.data.normalize import tokenize


def _write(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "tests" / "fixtures"))
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--distractors", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = load_dataset(cfg.dataset)
    rng = random.Random(cfg.seed)
    picked = rng.sample(sorted(r.id for r in ds.left), args.rows)
    left_by_id = {r.id: r for r in ds.left}

    right_tokens = {r.id: set(tokenize(r.name)) for r in ds.right}
    keep_right: set[str] = set()
    for lid in picked:
        keep_right |= ds.truth[lid]
        qt = set(tokenize(left_by_id[lid].name))
        scored = sorted(
            (
                (len(qt & toks) / max(1, len(qt | toks)), rid)
                for rid, toks in right_tokens.items()
                if rid not in ds.truth[lid]
            ),
            reverse=True,
        )
        keep_right |= {rid for _, rid in scored[: args.distractors]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = cfg.dataset
    left_rows = [left_by_id[i].fields for i in sorted(picked, key=int)]
    right_rows = [r.fields for r in ds.right if r.id in keep_right]
    mapping = [
        {d.mapping_left_col: lid, d.mapping_right_col: rid}
        for lid in sorted(picked, key=int)
        for rid in sorted(ds.truth[lid])
    ]
    _write(out / d.left_file, left_rows, list(left_rows[0]))
    _write(out / d.right_file, right_rows, list(right_rows[0]))
    _write(out / d.mapping_file, mapping, [d.mapping_left_col, d.mapping_right_col])
    print(f"fixture: {len(left_rows)} left, {len(right_rows)} right, {len(mapping)} pairs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
