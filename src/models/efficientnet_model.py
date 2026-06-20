from __future__ import annotations

import torch
from torch import nn

from .base_model import BaseModel


class EfficientNetModel(BaseModel):

    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__(num_classes)
        try:
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        except Exception as exc:
            raise RuntimeError("EfficientNet-B0 experiment requires torchvision to be installed.") from exc

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.network = efficientnet_b0(weights=weights)
        in_features = self.network.classifier[1].in_features
        self.network.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, images: torch.Tensor, layout_features: torch.Tensor | None = None) -> torch.Tensor:
        return self.network(images)
