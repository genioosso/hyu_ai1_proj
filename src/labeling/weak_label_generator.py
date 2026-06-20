from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from datasets.rico_parser import RicoParser
from features.layout_feature_extractor import LayoutFeatureExtractor

from .heuristic_usability_judge import HeuristicUsabilityJudge
from .usability_rubric import UsabilityAssessment, UsabilityRubric


class WeakLabelGenerator:

    def __init__(
        self,
        dataset_root: Path,
        labels_path: Path,
        rubric_path: Path,
        output_dir: Path,
        parser: RicoParser,
        layout_extractor: LayoutFeatureExtractor,
        strategy: str = "heuristic",
    ) -> None:
        self.dataset_root = dataset_root
        self.labels_path = labels_path
        self.rubric = UsabilityRubric.from_yaml(rubric_path)
        self.output_dir = output_dir
        self.parser = parser
        self.layout_extractor = layout_extractor
        self.strategy = strategy
        self.heuristic_judge = HeuristicUsabilityJudge(self.rubric, layout_extractor.feature_names)

    def generate_if_missing(self, overwrite: bool = False) -> Path:
        if self.labels_path.exists() and not overwrite:
            return self.labels_path
        sample_ids = self._discover_sample_ids()
        if not sample_ids:
            raise RuntimeError(
                "No paired Rico screenshots and hierarchy files were found. "
                "Place data under data/screenshots and data/hierarchies before generating labels."
        )
        assessments = [self._assess(sample_id) for sample_id in sample_ids]
        self._write_labels(assessments)
        self._write_assessments(assessments)
        return self.labels_path

    def _discover_sample_ids(self) -> list[str]:
        screenshot_dir = self.dataset_root / "screenshots"
        hierarchy_dir = self.dataset_root / "hierarchies"
        screenshot_ids = {path.stem for path in screenshot_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        hierarchy_ids = {path.stem for path in hierarchy_dir.glob("*.json")}
        return sorted(screenshot_ids & hierarchy_ids)

    def _assess(self, sample_id: str) -> UsabilityAssessment:
        screenshot_path = self._find_screenshot(sample_id)
        hierarchy_path = self.dataset_root / "hierarchies" / f"{sample_id}.json"
        components = self.parser.parse_file(hierarchy_path)
        with Image.open(screenshot_path) as image:
            width, height = image.size
        layout_features = self.layout_extractor.extract(components, width=width, height=height)
        return self.heuristic_judge.assess(sample_id, layout_features)

    def _find_screenshot(self, sample_id: str) -> Path:
        for suffix in (".jpg", ".jpeg", ".png"):
            path = self.dataset_root / "screenshots" / f"{sample_id}{suffix}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Screenshot not found for sample_id={sample_id}")

    def _write_labels(self, assessments: list[UsabilityAssessment]) -> None:
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        with self.labels_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "usability_score", "judge"])
            writer.writeheader()
            for assessment in assessments:
                writer.writerow(
                    {
                        "sample_id": assessment.sample_id,
                        "label": assessment.label,
                        "usability_score": assessment.usability_score,
                        "judge": assessment.judge,
                    }
                )

    def _write_assessments(self, assessments: list[UsabilityAssessment]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "usability_assessments.jsonl").open("w", encoding="utf-8") as handle:
            for assessment in assessments:
                handle.write(json.dumps(asdict(assessment), ensure_ascii=False) + "\n")
