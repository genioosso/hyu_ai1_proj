from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from labeling.w3c_rule_usability_judge import W3CUsabilityJudgment
from datasets.rico_parser import RicoParser
from features.layout_feature_extractor import LayoutFeatureExtractor
from standards.w3c_usability_builder import W3CUsabilitySpecBuilder
from utils.random_seed import set_seed

from .config import W3CRuleExperimentConfig
from .feature_pass import extract_feature_records
from .labeling_pass import make_w3c_labels
from .sample_picker import RicoSamplePicker
from .training_pass import train_models


class W3CRuleCVExperiment:

    def __init__(self, config: W3CRuleExperimentConfig) -> None:
        self.config = config
        set_seed(config.seed)

    def run(self) -> dict[str, Any]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # 2026.05.25, scyang, 실험 흐름은 여기서만 한눈에 보이게 묶음
        w3c_artifacts = W3CUsabilitySpecBuilder(self.config.project_root).build(refresh=False)
        parser = RicoParser()
        layout_extractor = LayoutFeatureExtractor()
        sample_picker = RicoSamplePicker(self.config.dataset_root)
        sample_ids = sample_picker.list_ids(self.config.sample_limit)

        # 2026.05.25, scyang, 모델 학습 전에 공통 특징부터 만들어둠
        feature_records = extract_feature_records(
            sample_ids=sample_ids,
            dataset_root=self.config.dataset_root,
            output_dir=self.config.output_dir,
            sample_picker=sample_picker,
            parser=parser,
            layout_extractor=layout_extractor,
            display_path=self._display_path,
        )

        # 2026.05.25, scyang, 규칙 라벨은 정답이라기보다 학습 재료에 가까움
        judgments, labels_path = make_w3c_labels(
            feature_records=feature_records,
            criteria_path=Path(w3c_artifacts["criteria"]),
            dataset_root=self.config.dataset_root,
            output_dir=self.config.output_dir,
            binary_threshold=self.config.binary_threshold,
            negative_ratio=self.config.negative_ratio,
            display_path=self._display_path,
        )

        # 2026.05.25, scyang, 모델 비교하려고 분할은 한 번만 정해서 씀
        metrics = train_models(
            labels_path=labels_path,
            config=self.config,
            parser=parser,
            layout_extractor=layout_extractor,
            display_path=self._display_path,
        )

        criterion_stats = self._criterion_stats(judgments)
        return self._write_run_metadata(
            w3c_artifacts=w3c_artifacts,
            feature_records=feature_records,
            judgments=judgments,
            metrics=metrics,
            criterion_stats=criterion_stats,
        )

    def _display_path(self, path: Path | str) -> str:
        resolved = Path(path)
        try:
            return resolved.relative_to(self.config.project_root).as_posix()
        except ValueError:
            return resolved.as_posix()

    def _criterion_stats(self, judgments: list[W3CUsabilityJudgment]) -> dict[str, Any]:
        criterion_values: dict[str, list[int]] = {}
        for judgment in judgments:
            for criterion, score in judgment.criterion_scores.items():
                criterion_values.setdefault(criterion, []).append(int(score))
        return {
            criterion: {
                "mean": sum(values) / max(len(values), 1),
                "min": min(values),
                "max": max(values),
                "distribution": dict(Counter(values)),
            }
            for criterion, values in criterion_values.items()
        }

    def _plot_inventory(self) -> list[str]:
        plot_dir = self.config.output_dir / "plots"
        if not plot_dir.exists():
            return []
        return [self._display_path(path) for path in sorted(plot_dir.glob("*.png"))]

    def _write_run_metadata(
        self,
        w3c_artifacts: dict[str, Path],
        feature_records: list[dict[str, Any]],
        judgments: list[W3CUsabilityJudgment],
        metrics: dict[str, dict[str, float]],
        criterion_stats: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = {
            "task": "Binary prediction of W3C-rule usability labels.",
            "dataset": "Rico UI screenshots and hierarchy metadata",
            "label_source": "W3C-inspired static UI rules",
            "setup": {
                "sample_limit": self.config.sample_limit,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "image_size": self.config.image_size,
                "device": self.config.device,
                "models": self.config.model_names,
                "seed": self.config.seed,
                "test_model": "best validation checkpoint",
            },
            "counts": {
                "features": len(feature_records),
                "labels": len(judgments),
                "label_distribution": dict(Counter(judgment.label for judgment in judgments)),
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
                "f1_definition": "positive-class F1 for the usable class",
            },
            "w3c_artifacts": {key: self._display_path(value) for key, value in w3c_artifacts.items()},
            "criterion_stats": criterion_stats,
            "model_metrics": metrics,
            "plots": self._plot_inventory(),
        }
        metadata_path = self.config.output_dir / "run_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return metadata
