"""Build the text that represents a record for retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from entity_cascade.schemas import Record


def record_text(rec: Record, fields: Sequence[str]) -> str:
    """Join the configured fields; fall back to the name if none of them is filled."""
    parts = [rec.fields.get(f, "").strip() for f in fields]
    text = " ".join(p for p in parts if p)
    return text or rec.name
