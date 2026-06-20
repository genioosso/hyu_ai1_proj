from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DataPreprocessor:

    image_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    horizontal_flip: bool = False

    def build_transform(self, training: bool = False):
        return ImageTransform(
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
            horizontal_flip=training and self.horizontal_flip,
        )


@dataclass(frozen=True)
class ImageTransform:

    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    horizontal_flip: bool = False

    def __call__(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize((self.image_size, self.image_size))
        if self.horizontal_flip:
            resized = resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std
