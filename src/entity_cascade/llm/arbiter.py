"""LLM arbiter: for gray-zone rows, ask the model to pick one of the top candidates or none."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from entity_cascade.config import LLMConfig
from entity_cascade.llm.base import LLMClient, LLMError
from entity_cascade.rerank.features import compute_features
from entity_cascade.schemas import ArbiterAnswer, Record

PROMPT_VERSIONS = ("v1", "v2", "v3", "v4")
_MAX_DESC = 300
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

INSTRUCTIONS = """\
You match supplier price-list items to a product catalog.
Decide which catalog candidate, if any, is exactly the same product as the price-list item.
Rules:
- The same product has the same brand and the same model number. Differences in colour,
  capacity, size or bundle contents (when they change the model number) mean a different product.
- Ignore wording differences, marketing text and price differences.
- If no candidate is clearly the same product, answer "none". Do not guess.
Answer with one JSON object and nothing else:
{"choice": <number 1..N or "none">, "confidence": <0.0..1.0>, "reason": "<short sentence>"}
"""

# v2: explicit decision procedure around the model number, and what confidence should mean.
INSTRUCTIONS_V2 = """\
You match supplier price-list items to a product catalog.
Decide which catalog candidate, if any, is exactly the same product as the price-list item.
How to decide:
1. Find the model number (letters and digits, e.g. PSLX350H or MDR-J10) in the price-list item.
2. A candidate with the same model number is the same product, ignoring case, dashes and spaces,
   and ignoring a colour or variant suffix such as /BLUE or BK/WH on an otherwise identical base
   model. Different wording, descriptions or prices do not matter.
3. If no candidate shares the model number, pick one only when brand and product type are the
   same and the names share distinctive words. Otherwise answer "none".
4. Never answer "none" only because the wording or the price differs.
Confidence: 0.9 or more when the model number matches; 0.6 to 0.8 when brand and product type
match but no model number can be compared; below 0.5 when you are unsure.
Answer with one JSON object and nothing else:
{"choice": <number 1..N or "none">, "confidence": <0.0..1.0>, "reason": "<short sentence>"}
"""

# v3: v2, but the model writes its comparison first and decides after it.
INSTRUCTIONS_V3 = INSTRUCTIONS_V2.replace(
    """Answer with one JSON object and nothing else:
{"choice": <number 1..N or "none">, "confidence": <0.0..1.0>, "reason": "<short sentence>"}
""",
    """Answer with one JSON object and nothing else. Write "reason" first: one sentence that
compares the model numbers of the item and the best candidate. Then decide.
{"reason": "<comparison>", "choice": <number 1..N or "none">, "confidence": <0.0..1.0>}
""",
)

# v4 = v3 plus a rule-based check line next to every candidate.
_VERSION_TEXT = {
    "v1": INSTRUCTIONS,
    "v2": INSTRUCTIONS_V2,
    "v3": INSTRUCTIONS_V3,
    "v4": INSTRUCTIONS_V3,
}


def _hint(row: Record, cand: Record) -> str:
    """Rule-based evidence shown next to a candidate in v4 (the signals the cascade uses)."""
    f = compute_features(row, cand)
    if f.code_exact:
        code = "model number equal"
    elif f.code_prefix:
        code = "model number is a prefix match"
    elif f.code_conflict:
        code = "model numbers differ"
    else:
        code = "no model number to compare"
    brand = (
        "brand equal" if f.brand_match else "brand differs" if f.brand_conflict else "brand unknown"
    )
    return f"check: {code}, {brand}"


def _fmt(rec: Record) -> str:
    desc = rec.description.strip().replace("\n", " ")
    if len(desc) > _MAX_DESC:
        desc = desc[:_MAX_DESC].rstrip() + "..."
    parts = [f"name: {rec.name}"]
    brand = rec.fields.get("manufacturer") or rec.brand
    if brand:
        parts.append(f"brand: {brand}")
    if desc:
        parts.append(f"description: {desc}")
    price = rec.fields.get("price", "").strip()
    if price:
        parts.append(f"price: {price}")
    return "; ".join(parts)


def build_prompt(row: Record, candidates: Sequence[Record], version: str = "v1") -> str:
    if version not in _VERSION_TEXT:
        raise ValueError(f"unknown prompt version {version!r}; known: {PROMPT_VERSIONS}")
    lines = [_VERSION_TEXT[version], "Price-list item:", _fmt(row), "", "Candidates:"]
    for i, c in enumerate(candidates, start=1):
        extra = f"; {_hint(row, c)}" if version == "v4" else ""
        lines.append(f"{i}. {_fmt(c)}{extra}")
    return "\n".join(lines) + "\n"


def parse_answer(text: str, n_candidates: int) -> ArbiterAnswer | None:
    """Parse and validate the model output. None if it is not a usable answer."""
    raw = _FENCE_RE.sub("", text.strip())
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        answer = ArbiterAnswer.model_validate(json.loads(raw[start : end + 1]))
    except (json.JSONDecodeError, ValidationError):
        return None
    if isinstance(answer.choice, int) and answer.choice > n_candidates:
        return None
    return answer


@dataclass(frozen=True)
class ArbiterResult:
    row_id: str
    shown_ids: tuple[str, ...]  # catalog ids of the candidates in prompt order
    answer: ArbiterAnswer | None  # None: provider failure or unusable output
    error: str | None
    input_tokens: int
    output_tokens: int
    latency_s: float

    @property
    def chosen_id(self) -> str | None:
        if self.answer is None or self.answer.choice == "none":
            return None
        return self.shown_ids[self.answer.choice - 1]


@dataclass(frozen=True)
class ArbiterItem:
    row: Record
    candidates: tuple[Record, ...]


async def _one(
    client: LLMClient, item: ArbiterItem, sem: asyncio.Semaphore, version: str
) -> ArbiterResult:
    prompt = build_prompt(item.row, item.candidates, version)
    shown = tuple(c.id for c in item.candidates)
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.complete(prompt)
        except LLMError as exc:  # provider down after retries: escalate this row, keep going
            return ArbiterResult(item.row.id, shown, None, str(exc), 0, 0, time.perf_counter() - t0)
    answer = parse_answer(resp.text, len(item.candidates))
    return ArbiterResult(
        row_id=item.row.id,
        shown_ids=shown,
        answer=answer,
        error=None if answer else "unusable model output",
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        latency_s=time.perf_counter() - t0,
    )


async def arbitrate(
    client: LLMClient, items: Sequence[ArbiterItem], cfg: LLMConfig
) -> list[ArbiterResult]:
    """Run all items concurrently, at most cfg.max_concurrency requests in flight.

    LLMError (provider failure) is caught per row. CassetteMiss and other bugs propagate.
    """
    sem = asyncio.Semaphore(cfg.max_concurrency)
    return list(await asyncio.gather(*(_one(client, it, sem, cfg.prompt_version) for it in items)))


def is_resolved(result: ArbiterResult, min_confidence: float) -> bool:
    return result.answer is not None and result.answer.confidence >= min_confidence
