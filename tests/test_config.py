from pathlib import Path

import pytest
from pydantic import ValidationError

from entity_cascade.config import AppConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.dataset.name == "abt_buy"
    assert cfg.retrieval.top_k == 20
    assert cfg.thresholds.target_precision == 0.98
    assert cfg.models.embedding == "BAAI/bge-small-en-v1.5"
    assert cfg.models.reranker == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_unknown_key_rejected() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    data = cfg.model_dump(mode="json")
    data["bogus"] = 1
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)
