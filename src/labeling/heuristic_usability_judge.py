from __future__ import annotations

from .usability_rubric import UsabilityAssessment, UsabilityRubric


class HeuristicUsabilityJudge:

    def __init__(self, rubric: UsabilityRubric, feature_names: tuple[str, ...]) -> None:
        self.rubric = rubric
        self.feature_names = feature_names

    def assess(self, sample_id: str, layout_features: list[float]) -> UsabilityAssessment:
        features = dict(zip(self.feature_names, layout_features))
        visual_clarity = self._score_visual_clarity(features)
        layout_consistency = self._score_layout_consistency(features)
        touch_target_accessibility = self._score_touch_targets(features)
        information_density = self._score_information_density(features)
        task_efficiency = self._score_task_efficiency(features)
        weighted_score = self._weighted_average(
            {
                "visual_clarity": visual_clarity,
                "layout_consistency": layout_consistency,
                "touch_target_accessibility": touch_target_accessibility,
                "information_density": information_density,
                "task_efficiency_estimate": task_efficiency,
            }
        )
        usability_score = self._clamp(round(weighted_score))
        return UsabilityAssessment(
            sample_id=sample_id,
            label=self.rubric.to_label(usability_score),
            usability_score=usability_score,
            visual_clarity=visual_clarity,
            layout_consistency=layout_consistency,
            touch_target_accessibility=touch_target_accessibility,
            information_density=information_density,
            task_efficiency_estimate=task_efficiency,
            judge="heuristic_rubric",
            rationale="Weak label generated from UI usability rubric thresholds and Rico layout metadata.",
        )

    def _score_visual_clarity(self, features: dict[str, float]) -> int:
        whitespace = features.get("whitespace_ratio", 0.0)
        complexity = features.get("complexity_score", 0.0)
        if whitespace < self._threshold("low_whitespace") or complexity > self._threshold("high_complexity"):
            return 2
        return 4

    def _score_layout_consistency(self, features: dict[str, float]) -> int:
        component_density = features.get("component_density", 0.0)
        return 2 if component_density > self._threshold("high_density") else 4

    def _score_touch_targets(self, features: dict[str, float]) -> int:
        button_ratio = features.get("button_ratio", 0.0)
        input_ratio = features.get("input_ratio", 0.0)
        return 4 if 0.05 <= button_ratio + input_ratio <= 0.45 else 3

    def _score_information_density(self, features: dict[str, float]) -> int:
        text_ratio = features.get("text_ratio", 0.0)
        component_density = features.get("component_density", 0.0)
        if text_ratio > self._threshold("excessive_text_ratio") or component_density > self._threshold("high_density"):
            return 2
        return 4

    def _score_task_efficiency(self, features: dict[str, float]) -> int:
        complexity = features.get("complexity_score", 0.0)
        input_count = features.get("input_count", 0.0)
        return 2 if complexity > self._threshold("high_complexity") and input_count > 3 else 4

    def _weighted_average(self, scores: dict[str, int]) -> float:
        total_weight = sum(self.rubric.criteria_weights.get(name, 0.0) for name in scores)
        if total_weight <= 0:
            return sum(scores.values()) / max(len(scores), 1)
        return sum(scores[name] * self.rubric.criteria_weights.get(name, 0.0) for name in scores) / total_weight

    def _threshold(self, name: str) -> float:
        return float(self.rubric.heuristic_thresholds.get(name, 0.0))

    def _clamp(self, score: int) -> int:
        return max(self.rubric.minimum_score, min(self.rubric.maximum_score, score))
