from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from features.layout_feature_extractor import LayoutFeatureExtractor

from .rico_parser import RicoParser


@dataclass(frozen=True)
class UISample:

    sample_id: str
    screenshot_path: Path
    hierarchy_path: Path
    label: int


class UIUsabilityDataset(Dataset[dict[str, Any]]):

    def __init__(
        self,
        samples: list[UISample],
        parser: RicoParser,
        layout_extractor: LayoutFeatureExtractor,
        image_transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.samples = samples
        self.parser = parser
        self.layout_extractor = layout_extractor
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample.screenshot_path).convert("RGB")
        components = self.parser.parse_file(sample.hierarchy_path)
        width, height = image.size
        layout_features = self.layout_extractor.extract(components, width=width, height=height)
        return {
            "sample_id": sample.sample_id,
            "image": self.image_transform(image),
            "layout": torch.tensor(layout_features, dtype=torch.float32),
            "label": torch.tensor(sample.label, dtype=torch.long),
        }


class CachedUIUsabilityDataset(UIUsabilityDataset):

    def __init__(
        self,
        samples: list[UISample],
        parser: RicoParser,
        layout_extractor: LayoutFeatureExtractor,
        image_transform: Callable[[Image.Image], torch.Tensor],
        cache_dir: Path,
        cache_namespace: str,
    ) -> None:
        super().__init__(samples, parser, layout_extractor, image_transform)
        self.cache_dir = cache_dir / cache_namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        cache_path = self.cache_dir / f"{sample.sample_id}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu")
            return {
                "sample_id": sample.sample_id,
                "image": cached["image"],
                "layout": cached["layout"],
                "label": torch.tensor(sample.label, dtype=torch.long),
            }

        item = super().__getitem__(index)
        torch.save({"image": item["image"], "layout": item["layout"]}, cache_path)
        return item
