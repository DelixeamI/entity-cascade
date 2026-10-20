"""Everything fitted from labels, in one place. Sees val only.

`calibrate` accepts a ValSplit and val rankings inputs; there is no code path from here to the
test split. tests/test_no_leakage.py checks that, by type, by behaviour and by source scan.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from entity_cascade.config import AppConfig
from entity_cascade.data.split import ValSplit
from entity_cascade.rerank.features import FeatureWeights
from entity_cascade.rerank.ranker import RawRow, rank_row
from entity_cascade.rerank.weights import fit_feature_weights
from entity_cascade.zones.assign import Thresholds
from entity_cascade.zones.thresholds import fit_thresholds


class Calibration(BaseModel):
    model_config = ConfigDict(frozen=True)

    weights: FeatureWeights
    thresholds: Thresholds

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Calibration:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def calibrate(val: ValSplit, val_raws: Mapping[str, RawRow], cfg: AppConfig) -> Calibration:
    weights = fit_feature_weights(val, val_raws, cfg.rerank.weight_grid)
    rankings = {r.id: rank_row(val_raws[r.id], weights) for r in val.rows}
    thresholds = fit_thresholds(val, rankings, cfg.thresholds)
    return Calibration(weights=weights, thresholds=thresholds)
