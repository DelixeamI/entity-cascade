"""Shared Pydantic models."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    """One product row (a price-list line or a catalog item)."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str = ""
    brand: str | None = None
    fields: dict[str, str] = Field(default_factory=dict)


class Zone(StrEnum):
    AUTO_MATCH = "auto_match"
    LLM_ARBITER = "llm_arbiter"
    HUMAN_REVIEW = "human_review"
    NO_MATCH = "no_match"


class ArbiterAnswer(BaseModel):
    """Structured LLM answer. `choice` is the number of a shown candidate (1-based, as printed
    in the prompt) or "none" when no candidate is the same product."""

    model_config = ConfigDict(frozen=True)

    choice: Annotated[int, Field(ge=1)] | Literal["none"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
