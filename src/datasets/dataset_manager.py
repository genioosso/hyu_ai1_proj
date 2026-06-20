from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from features.layout_feature_extractor import LayoutFeatureExtractor

from .rico_parser import RicoParser
from .ui_dataset import CachedUIUsabilityDataset, UISample, UIUsabilityDataset


@dataclass(frozen=True)
class DatasetSplit:

    train: list[UISample]
    validation: list[UISample]
    test: list[UISample]


class DatasetManager:

    def __init__(
        self,
        dataset_root: Path,
        labels_path: Path,
        parser: RicoParser,
        layout_extractor: LayoutFeatureExtractor,
        seed: int = 42,
    ) -> None:
        self.dataset_root = dataset_root
        self.labels_path = labels_path
        self.parser = parser
        self.layout_extractor = layout_extractor
        self.seed = seed

    def load_samples(self) -> list[UISample]:
        samples: list[UISample] = []
        missing_hierarchies: list[str] = []
        with self.labels_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sample_id = row["sample_id"]
                hierarchy_path = self.dataset_root / "hierarchies" / f"{sample_id}.json"
                # 2026.05.18, hhlee, 파일 빠진 샘플은 여기서 바로 걸러야 뒤가 편함
                if not hierarchy_path.exists():
                    missing_hierarchies.append(sample_id)
                    continue
                samples.append(
                    UISample(
                        sample_id=sample_id,
                        screenshot_path=self._find_screenshot(sample_id),
                        hierarchy_path=hierarchy_path,
                        label=self._read_label(row),
                    )
                )
        if missing_hierarchies:
            preview = ", ".join(missing_hierarchies[:5])
            raise FileNotFoundError(f"Hierarchy JSON missing for {len(missing_hierarchies)} samples: {preview}")
        if not samples:
            raise ValueError(f"No usable Rico samples loaded from {self.labels_path}")
        return samples

    def _read_label(self, row: dict[str, str]) -> int:
        label = int(row["label"])
        # 2026.05.25, scyang, 라벨 값 이상하면 학습 결과 믿기 어려움
        if label not in (0, 1):
            raise ValueError(f"Expected binary label 0/1 for sample_id={row.get('sample_id')}, got {label}")
        return label

    def _find_screenshot(self, sample_id: str) -> Path:
        for suffix in (".jpg", ".jpeg", ".png"):
            path = self.dataset_root / "screenshots" / f"{sample_id}{suffix}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Screenshot not found for sample_id={sample_id}")

    def split(self, samples: list[UISample], train_ratio: float, validation_ratio: float) -> DatasetSplit:
        shuffled = list(samples)
        # 2026.06.02, hhlee, 파일 순서 말고 난수값으로만 분할 고정함
        random.Random(self.seed).shuffle(shuffled)
        train_end = int(len(shuffled) * train_ratio)
        validation_end = train_end + int(len(shuffled) * validation_ratio)
        return DatasetSplit(
            train=shuffled[:train_end],
            validation=shuffled[train_end:validation_end],
            test=shuffled[validation_end:],
        )

    def create_loader(
        self,
        samples: list[UISample],
        image_transform,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
        cache_dir: Path | None = None,
        cache_namespace: str = "default",
    ) -> DataLoader:
        if cache_dir is None:
            dataset = UIUsabilityDataset(samples, self.parser, self.layout_extractor, image_transform)
        else:
            dataset = CachedUIUsabilityDataset(
                samples,
                self.parser,
                self.layout_extractor,
                image_transform,
                cache_dir=cache_dir,
                cache_namespace=cache_namespace,
            )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
