"""Cheap, explainable match features: model-code and brand agreement."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from entity_cascade.data.normalize import extract_codes, normalize_brand
from entity_cascade.schemas import Record

MIN_PREFIX_LEN = 5


class MatchFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    code_exact: bool  # some model code is equal on both sides
    code_prefix: bool  # one code is a prefix of another (MDRJ10 vs MDRJ10BLUE), not equal
    code_conflict: bool  # both sides have codes, none agree
    brand_match: bool
    brand_conflict: bool  # both sides have a brand and they differ


class FeatureWeights(BaseModel):
    """Additive adjustments to the cross-encoder logit. Fitted on val only."""

    model_config = ConfigDict(frozen=True)

    code_exact: float = 0.0
    code_prefix: float = 0.0
    code_conflict: float = 0.0  # applied with a minus sign
    brand_match: float = 0.0
    brand_conflict: float = 0.0  # applied with a minus sign

    def adjust(self, f: MatchFeatures) -> float:
        return (
            self.code_exact * f.code_exact
            + self.code_prefix * f.code_prefix
            - self.code_conflict * f.code_conflict
            + self.brand_match * f.brand_match
            - self.brand_conflict * f.brand_conflict
        )


def record_codes(rec: Record) -> set[str]:
    """Model codes from the name and the description."""
    return set(extract_codes(rec.name)) | set(extract_codes(rec.description))


def _prefix_related(a: str, b: str) -> bool:
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= MIN_PREFIX_LEN and short != long_ and long_.startswith(short)


def compute_features(left: Record, right: Record) -> MatchFeatures:
    lc, rc = record_codes(left), record_codes(right)
    exact = bool(lc & rc)
    prefix = not exact and any(_prefix_related(a, b) for a in lc for b in rc)
    lb = normalize_brand(left.brand) if left.brand else ""
    rb = normalize_brand(right.brand) if right.brand else ""
    return MatchFeatures(
        code_exact=exact,
        code_prefix=prefix,
        code_conflict=bool(lc and rc) and not exact and not prefix,
        brand_match=bool(lb and rb and lb == rb),
        brand_conflict=bool(lb and rb and lb != rb),
    )


def feature_vector(f: MatchFeatures) -> Sequence[float]:
    """Same order as FeatureWeights.adjust, signs included, for vectorized grid search."""
    return (
        float(f.code_exact),
        float(f.code_prefix),
        -float(f.code_conflict),
        float(f.brand_match),
        -float(f.brand_conflict),
    )
