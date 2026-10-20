"""LLM client interface, shared types and the cassette key."""

from __future__ import annotations

import hashlib
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class LLMError(Exception):
    """Transport or provider failure after all retries. The row falls back to a human."""


class CassetteMiss(Exception):
    """RecordedLLM has no cassette for this (model, prompt). Run `make record`."""


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    async def complete(self, prompt: str) -> LLMResponse: ...


def cassette_key(model: str, prompt: str) -> str:
    """sha256 over model and prompt, NUL-separated so ("ab", "c") != ("a", "bc")."""
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()
