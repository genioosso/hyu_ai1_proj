from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets.rico_parser import UIComponent


@dataclass(frozen=True)
class LayoutFeatureExtractor:

    feature_names: tuple[str, ...] = (
        "button_count",
        "text_count",
        "image_count",
        "input_count",
        "layout_density",
        "whitespace_ratio",
        "component_density",
        "complexity_score",
        "button_ratio",
        "text_ratio",
        "image_ratio",
        "input_ratio",
    )

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def extract(self, components: list[UIComponent], width: int, height: int) -> list[float]:
        counts = Counter(component.component_type for component in components)
        total = max(len(components), 1)
        screen_area = max(width * height, 1)
        occupied_area = sum(self._area(component.bounds) for component in components)
        layout_density = min(occupied_area / screen_area, 1.0)
        whitespace_ratio = max(1.0 - layout_density, 0.0)
        component_density = total / screen_area
        # 2026.06.02, scyang, 조작 요소가 많으면 복잡한 화면으로 더 세게 잡음
        complexity_score = (
            counts["button"] * 1.2
            + counts["text"] * 0.8
            + counts["image"] * 1.0
            + counts["input"] * 1.5
            + len(set(counts)) * 0.5
        ) / total
        return [
            float(counts["button"]),
            float(counts["text"]),
            float(counts["image"]),
            float(counts["input"]),
            float(layout_density),
            float(whitespace_ratio),
            float(component_density),
            float(complexity_score),
            float(counts["button"] / total),
            float(counts["text"] / total),
            float(counts["image"] / total),
            float(counts["input"] / total),
        ]

    def _area(self, bounds: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bounds
        return max(x2 - x1, 0) * max(y2 - y1, 0)
