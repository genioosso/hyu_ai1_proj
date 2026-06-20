from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from datasets.rico_parser import RicoParser
from features.layout_feature_extractor import LayoutFeatureExtractor


@dataclass(frozen=True)
class UISampleFeatures:

    sample_id: str
    screenshot_path: str
    hierarchy_path: str
    image: dict[str, float]
    layout_features: dict[str, float]
    component_counts: dict[str, int]
    hierarchy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SampleFeatureExtractor:

    def __init__(self, parser: RicoParser, layout_extractor: LayoutFeatureExtractor) -> None:
        self.parser = parser
        self.layout_extractor = layout_extractor

    def extract(self, sample_id: str, screenshot_path: Path, hierarchy_path: Path) -> UISampleFeatures:
        components = self.parser.parse_file(hierarchy_path)
        counts = Counter(component.component_type for component in components)
        with Image.open(screenshot_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            image_features = self._image_features(rgb)
        layout_values = self.layout_extractor.extract(components, width=width, height=height)
        layout_features = dict(zip(self.layout_extractor.feature_names, layout_values))
        text_components = sum(1 for component in components if component.text.strip())
        clickable_components = sum(1 for component in components if component.clickable)
        enabled_components = sum(1 for component in components if component.enabled)
        return UISampleFeatures(
            sample_id=sample_id,
            screenshot_path=str(screenshot_path),
            hierarchy_path=str(hierarchy_path),
            image=image_features,
            layout_features={key: float(value) for key, value in layout_features.items()},
            component_counts={key: int(value) for key, value in counts.items()},
            hierarchy={
                "component_total": len(components),
                "text_component_count": text_components,
                "clickable_component_count": clickable_components,
                "enabled_component_count": enabled_components,
                "screen_width": width,
                "screen_height": height,
            },
        )

    def write_jsonl(self, records: list[UISampleFeatures], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return output_path

    def _image_features(self, image: Image.Image) -> dict[str, float]:
        analysis_image = image.copy()
        # 2026.06.02, scyang, 전체 이미지는 무겁고 여기선 대략 통계만 필요함
        analysis_image.thumbnail((160, 160))
        array = np.asarray(analysis_image, dtype=np.float32) / 255.0
        grayscale_image = analysis_image.convert("L")
        grayscale = np.asarray(grayscale_image, dtype=np.float32) / 255.0
        edges = np.asarray(grayscale_image.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
        return {
            "brightness_mean": float(grayscale.mean()),
            "brightness_std": float(grayscale.std()),
            "contrast": float(grayscale.max() - grayscale.min()),
            "edge_density": float((edges > 0.2).mean()),
            "red_mean": float(array[:, :, 0].mean()),
            "green_mean": float(array[:, :, 1].mean()),
            "blue_mean": float(array[:, :, 2].mean()),
        }
