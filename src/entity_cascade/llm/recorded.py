"""Cassettes: record real LLM answers once, replay them in tests and CI without keys."""

from __future__ import annotations

import json
from pathlib import Path

from entity_cascade.llm.base import CassetteMiss, LLMClient, LLMResponse, cassette_key


def _path(directory: Path, key: str) -> Path:
    return directory / f"{key}.json"


class RecordedLLM:
    """Replays cassettes. A missing cassette is an error, never a made-up answer."""

    def __init__(self, model: str, directory: Path) -> None:
        self._model = model
        self._dir = directory

    @property
    def model(self) -> str:
        return self._model

    def has(self, prompt: str) -> bool:
        return _path(self._dir, cassette_key(self._model, prompt)).exists()

    async def complete(self, prompt: str) -> LLMResponse:
        key = cassette_key(self._model, prompt)
        path = _path(self._dir, key)
        if not path.exists():
            raise CassetteMiss(
                f"no cassette {key[:12]}... for model {self._model!r} in {self._dir}; "
                "run `make record` with a real provider"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse.model_validate(data["response"])


class RecordingLLM:
    """Wraps a real client and writes every answer to a cassette (`make record`)."""

    def __init__(self, inner: LLMClient, directory: Path) -> None:
        self._inner = inner
        self._dir = directory
        directory.mkdir(parents=True, exist_ok=True)

    @property
    def model(self) -> str:
        return self._inner.model

    async def complete(self, prompt: str) -> LLMResponse:
        response = await self._inner.complete(prompt)
        key = cassette_key(self._inner.model, prompt)
        payload = {
            "model": self._inner.model,
            "prompt": prompt,
            "response": response.model_dump(),
        }
        _path(self._dir, key).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        return response
