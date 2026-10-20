"""Download a benchmark archive, verify its sha256 and extract it.

The expected hash is pinned in data/checksums.json. On the first download of a dataset
the hash is recorded there (trust on first use); every later download must match it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import httpx

from entity_cascade.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CHECKSUMS = ROOT / "data" / "checksums.json"


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def load_checksums(path: Path = CHECKSUMS) -> dict[str, str]:
    if not path.exists():
        return {}
    loaded: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def verify_or_pin(name: str, digest: str, path: Path = CHECKSUMS) -> bool:
    """Return True if pinned and matching, or newly pinned. False on mismatch."""
    known = load_checksums(path)
    if name in known:
        return known[name] == digest
    known[name] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(known, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"pinned sha256 for {name}: {digest}")
    return True


def extract_flat(blob: bytes, target: Path) -> list[str]:
    """Extract all CSV files from the archive directly into target (no subfolders)."""
    import io

    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name
            if info.is_dir() or not base or base.startswith("."):
                continue
            if "__MACOSX" in info.filename:
                continue
            (target / base).write_bytes(zf.read(info))
            written.append(base)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    ds = cfg.dataset

    print(f"downloading {ds.archive_url}")
    resp = httpx.get(ds.archive_url, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()
    digest = sha256_bytes(resp.content)
    if not verify_or_pin(ds.name, digest):
        print(f"sha256 mismatch for {ds.name}: got {digest}", file=sys.stderr)
        return 1
    target = ROOT / ds.data_dir
    files = extract_flat(resp.content, target)
    print(f"extracted {len(files)} files to {target}: {files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
