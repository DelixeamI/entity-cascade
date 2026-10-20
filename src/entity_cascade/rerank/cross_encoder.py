"""Cross-encoder reranker (CPU)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class Reranker(Protocol):
    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Relevance score per (query, document) pair. Higher is more relevant."""
        ...


class CachedReranker:
    """Disk cache around a reranker. Speed only: keys are content hashes of model + pair, so a
    cache hit returns exactly what the model returned before. Not used for latency numbers."""

    def __init__(self, inner: Reranker, model_name: str, path: Path) -> None:
        self._inner = inner
        self._model = model_name
        self._path = path
        self._cache: dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    rec = json.loads(line)
                    self._cache[rec["k"]] = float(rec["s"])

    def _key(self, query: str, doc: str) -> str:
        h = hashlib.sha256()
        for part in (self._model, query, doc):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        keys = [self._key(q, d) for q, d in pairs]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        self.hits += len(pairs) - len(missing)
        self.misses += len(missing)
        if missing:
            fresh = self._inner.score([pairs[i] for i in missing])
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                for i, s in zip(missing, fresh, strict=True):
                    if keys[i] not in self._cache:
                        self._cache[keys[i]] = s
                        f.write(json.dumps({"k": keys[i], "s": s}) + "\n")
        return [self._cache[k] for k in keys]


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder returning raw logits.

    Raw logits are used instead of sigmoid probabilities: a sigmoid saturates near 0 and 1 and
    would make the margin between two confident candidates vanish.
    """

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device="cpu")
        self._batch_size = batch_size
        self._identity = torch.nn.Identity()

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        out = self._model.predict(
            [list(p) for p in pairs],
            batch_size=self._batch_size,
            activation_fn=self._identity,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(x) for x in out.reshape(-1)]
