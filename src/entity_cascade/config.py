"""Typed configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: Path) -> Path:
    """Resolve a config path relative to the project root unless it is absolute."""
    return path if path.is_absolute() else PROJECT_ROOT / path


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_Cfg):
    name: str
    archive_url: str
    data_dir: Path
    left_file: str
    right_file: str
    mapping_file: str
    left_id: str
    right_id: str
    mapping_left_col: str
    mapping_right_col: str
    left_name: str
    right_name: str
    left_text: list[str]
    right_text: list[str]
    right_brand: str | None = None
    left_brand: str | None = None
    new_product_fraction: float = Field(default=0.0, ge=0.0, le=1.0)


class ModelsConfig(_Cfg):
    embedding: str
    reranker: str
    query_instruction: str = ""


class RetrievalConfig(_Cfg):
    top_k: int = 20
    per_mode_k: int = 50
    rrf_k: int = 60
    qdrant_url: str = ":memory:"
    collection: str = "catalog"


class RerankConfig(_Cfg):
    batch_size: int = Field(default=32, ge=1)
    # Fields of the price-list row sent to the cross-encoder as the query. The long Abt
    # description hurts it: on val, name-only queries beat name+description (DESIGN.md).
    query_fields: list[str] = ["name"]
    # Candidate values for each feature weight; the best combination is chosen on val only.
    weight_grid: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0)
    # Disk cache of cross-encoder scores (speed only). None disables it.
    cache_path: Path | None = Path(".cache/rerank_scores.jsonl")


class ThresholdsConfig(_Cfg):
    # auto_match: precision the zone must reach on val, and the minimum rows that back it.
    target_precision: float = Field(default=0.98, gt=0.0, le=1.0)
    min_auto_support: int = Field(default=20, ge=1)
    # Lower confidence bound for the precision estimate: 0.0 = plain empirical precision,
    # 1.96 = Wilson 95% lower bound (much more conservative on small val sets).
    wilson_z: float = Field(default=0.0, ge=0.0)
    # llm_arbiter band: its lower edge is the lowest score at which at least this share of
    # rows is "answerable": a true item is among the candidates shown to the arbiter, or the
    # row truly has no pair (so the right answer is "none"). Rows whose true item exists but
    # is not shown cannot be answered correctly and are better sent to a human.
    llm_min_answerable: float = Field(default=0.5, ge=0.0, le=1.0)
    min_llm_support: int = Field(default=10, ge=1)
    # no_match: share of the zone that must truly have no pair, and the minimum rows.
    no_match_target_precision: float = Field(default=0.95, gt=0.0, le=1.0)
    min_no_match_support: int = Field(default=10, ge=1)


class LLMConfig(_Cfg):
    provider: Literal["fake", "recorded", "openai"] = "recorded"
    base_url: str = "http://localhost:11434/v1"
    model: str = "unset"
    api_key_env: str = "LLM_API_KEY"
    cassette_dir: Path = Path("cassettes/eval")
    max_concurrency: int = Field(default=8, ge=1)
    timeout_s: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=4, ge=0)
    backoff_base_s: float = Field(default=1.0, gt=0.0)
    top_candidates: int = Field(default=3, ge=1)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    # Chosen on val (reports/results.md, "LLM prompt tuning"); v1 is the first draft.
    prompt_version: str = "v1"
    # Val tuning rule: resolve as many rows as possible while the arbiter's answers on the rows
    # it resolves are at least this accurate; min_resolved_support guards against tiny sets.
    target_accuracy: float = Field(default=0.90, gt=0.0, le=1.0)
    min_resolved_support: int = Field(default=10, ge=1)
    price_input_per_1m: float = Field(default=0.0, ge=0.0)
    price_output_per_1m: float = Field(default=0.0, ge=0.0)


class AppConfig(_Cfg):
    seed: int = 13
    val_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    dataset: DatasetConfig
    models: ModelsConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    rerank: RerankConfig = RerankConfig()
    thresholds: ThresholdsConfig = ThresholdsConfig()
    llm: LLMConfig = LLMConfig()


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open(encoding="utf-8-sig") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
