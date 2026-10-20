"""Test doubles for LLMClient. Neither is a real model: never quote them as LLM quality."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from entity_cascade.data.normalize import extract_codes, tokenize
from entity_cascade.llm.base import LLMResponse


def _tokens(text: str) -> int:
    return len(text.split())


class FakeLLM:
    """Answers with whatever `responder(prompt)` returns. Default: always candidate 1."""

    def __init__(
        self,
        responder: Callable[[str], str] | None = None,
        model: str = "fake",
    ) -> None:
        self._responder = responder or (
            lambda _p: json.dumps({"choice": 1, "confidence": 0.9, "reason": "fake"})
        )
        self._model = model
        self.prompts: list[str] = []

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        text = self._responder(prompt)
        return LLMResponse(text=text, input_tokens=_tokens(prompt), output_tokens=_tokens(text))


_ITEM_RE = re.compile(r"Price-list item:\n(?P<item>.*?)\n\nCandidates:\n(?P<cands>.*)", re.DOTALL)
_CAND_RE = re.compile(r"^\d+\. (.*)$", re.MULTILINE)


def heuristic_responder(prompt: str) -> str:
    """Pick the candidate with the best token overlap, requiring a shared model code or
    a high overlap. Deterministic and cheap: stands in for an LLM where plumbing is tested."""
    m = _ITEM_RE.search(prompt)
    if m is None:
        return json.dumps({"choice": "none", "confidence": 0.0, "reason": "unparseable"})
    item = m.group("item")
    cands = _CAND_RE.findall(m.group("cands"))
    item_tokens, item_codes = set(tokenize(item)), set(extract_codes(item))
    best_i, best = -1, 0.0
    for i, c in enumerate(cands):
        c_tokens, c_codes = set(tokenize(c)), set(extract_codes(c))
        jac = len(item_tokens & c_tokens) / max(1, len(item_tokens | c_tokens))
        score = jac + (1.0 if item_codes & c_codes else 0.0)
        if score > best:
            best_i, best = i, score
    if best_i == -1 or best < 0.5:
        return json.dumps({"choice": "none", "confidence": 0.8, "reason": "no close candidate"})
    confidence = min(0.95, 0.5 + best / 2)
    return json.dumps(
        {"choice": best_i + 1, "confidence": round(confidence, 2), "reason": "overlap"}
    )


class HeuristicFakeLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__(heuristic_responder, model="fake-heuristic-v1")
