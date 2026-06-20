from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class W3CUsabilityJudgment:

    sample_id: str
    label: int
    usability_score: int
    criterion_scores: dict[str, int]
    uncertainty: float
    judge: str
    rationale: str
    unobservable_items: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class W3CRuleUsabilityJudge:

    OBSERVABLE_CRITERIA = (
        "visual_perceivability",
        "operable_navigation",
        "understandable_labels",
        "robust_semantics",
    )

    def __init__(self, criteria_path: Path, binary_threshold: int = 3) -> None:
        self.criteria_path = criteria_path
        self.binary_threshold = binary_threshold
        self.criteria = json.loads(criteria_path.read_text(encoding="utf-8")).get("criteria", [])

    def assess(self, feature_record: dict[str, Any], _screenshot_path: Path | None = None) -> W3CUsabilityJudgment:
        layout = feature_record["layout_features"]
        image = feature_record["image"]
        hierarchy = feature_record["hierarchy"]
        density = float(layout.get("component_density", 0.0))
        complexity = float(layout.get("complexity_score", 0.0))
        whitespace = float(layout.get("whitespace_ratio", 1.0))
        text_ratio = float(layout.get("text_ratio", 0.0))
        layout_density = float(layout.get("layout_density", 0.0))
        edge_density = float(image.get("edge_density", 0.0))
        clickable = float(hierarchy.get("clickable_component_count", 0))
        total = max(float(hierarchy.get("component_total", 1)), 1.0)
        clickable_ratio = clickable / total

        # 2026.06.01, scyang, 실제 실행 없이 확인 가능한 것만 규칙에 넣음
        visual = self._score(
            5.0
            - edge_density * 8.0
            - max(layout_density - 0.82, 0) * 5.0
            - max(0.22 - whitespace, 0) * 5.0
        )
        operable = self._score(
            4.4
            - max(density - 0.000055, 0) * 9000
            - max(clickable_ratio - 0.42, 0) * 4.0
        )
        understandable = self._score(
            4.6
            - max(text_ratio - 0.35, 0) * 6.0
            - max(complexity - 1.05, 0) * 3.0
            - max(total - 35, 0) * 0.04
        )
        robust = self._score(4.0 - max(total - 50, 0) * 0.03 + min(clickable, 6) * 0.08)
        criterion_scores = {
            "visual_perceivability": visual,
            "operable_navigation": operable,
            "understandable_labels": understandable,
            "robust_semantics": robust,
        }
        mean_score = sum(criterion_scores.values()) / len(criterion_scores)
        # 2026.05.22, hhlee, 한 항목이 크게 나쁘면 평균으로 덮지 못하게 함
        usability_score = self._score(min(mean_score, min(criterion_scores.values()) + 1.0))
        return W3CUsabilityJudgment(
            sample_id=str(feature_record["sample_id"]),
            label=int(usability_score >= self.binary_threshold),
            usability_score=usability_score,
            criterion_scores=criterion_scores,
            uncertainty=0.35,
            judge="w3c_rule",
            rationale=(
                "Rule score from static UI signals: edge density, whitespace, layout density, "
                "text ratio, component density, clickable ratio, complexity, and component count."
            ),
            unobservable_items=[
                "runtime behavior",
                "keyboard navigation",
                "assistive technology behavior",
                "task success",
                "user satisfaction",
            ],
        )

    def write_jsonl(self, judgments: list[W3CUsabilityJudgment], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for judgment in judgments:
                handle.write(json.dumps(judgment.to_dict(), ensure_ascii=False) + "\n")
        return output_path

    def labels_rows(self, judgments: list[W3CUsabilityJudgment]) -> list[dict[str, Any]]:
        return [
            {
                "sample_id": judgment.sample_id,
                "label": judgment.label,
                "usability_score": judgment.usability_score,
                "judge": judgment.judge,
            }
            for judgment in judgments
        ]

    def _score(self, raw_value: float) -> int:
        return max(1, min(5, int(round(raw_value))))
