from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class UsabilityAssessment:

    sample_id: str
    label: int
    usability_score: int
    visual_clarity: int
    layout_consistency: int
    touch_target_accessibility: int
    information_density: int
    task_efficiency_estimate: int
    judge: str
    rationale: str


@dataclass(frozen=True)
class UsabilityRubric:

    binary_threshold: int
    minimum_score: int
    maximum_score: int
    criteria_weights: dict[str, float]
    heuristic_thresholds: dict[str, float]

    @classmethod
    def from_yaml(cls, path: Path) -> "UsabilityRubric":
        with path.open("r", encoding="utf-8") as handle:
            payload: dict[str, Any] = yaml.safe_load(handle) or {}
        criteria = payload.get("criteria", {})
        thresholds = payload.get("heuristic_thresholds", {})
        labeling = payload.get("labeling", {})
        return cls(
            binary_threshold=int(labeling.get("binary_threshold", 3)),
            minimum_score=int(thresholds.get("minimum_score", 1)),
            maximum_score=int(thresholds.get("maximum_score", 5)),
            criteria_weights={name: float(data.get("weight", 0.0)) for name, data in criteria.items()},
            heuristic_thresholds={name: float(value) for name, value in thresholds.items()},
        )

    def to_label(self, usability_score: int) -> int:
        return int(usability_score >= self.binary_threshold)
