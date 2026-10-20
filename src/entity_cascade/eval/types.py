"""Shared pydantic types for the eval results."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator


def _null_to_nan(value: Any) -> Any:
    """JSON has no NaN: pydantic writes it as null, so read null back as NaN."""
    return float("nan") if value is None else value


NanFloat = Annotated[float, BeforeValidator(_null_to_nan)]
