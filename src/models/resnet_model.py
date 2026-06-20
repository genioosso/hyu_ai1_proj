from __future__ import annotations

import torch
from torch import nn

from .base_model import BaseModel


class ResNetModel(BaseModel):

    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__(num_classes)
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:
            raise RuntimeError("ResNet18 experiment requires torchvision to be installed.") from exc

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.network = resnet18(weights=weights)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, images: torch.Tensor, layout_features: torch.Tensor | None = None) -> torch.Tensor:
        return self.network(images)
