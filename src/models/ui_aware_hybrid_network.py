from __future__ import annotations

import torch
from torch import nn

from features.image_feature_extractor import ImageFeatureExtractor

from .base_model import BaseModel


class UIAwareHybridNetwork(BaseModel):

    def __init__(
        self,
        num_classes: int,
        layout_feature_dim: int,
        image_feature_dim: int = 128,
        layout_hidden_dim: int = 64,
        fusion_hidden_dim: int = 128,
    ) -> None:
        super().__init__(num_classes)
        self.image_branch = ImageFeatureExtractor(output_dim=image_feature_dim)
        self.layout_branch = nn.Sequential(
            nn.Linear(layout_feature_dim, layout_hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(layout_hidden_dim),
            nn.Dropout(0.2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(image_feature_dim + layout_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor, layout_features: torch.Tensor | None = None) -> torch.Tensor:
        if layout_features is None:
            raise ValueError("UIAwareHybridNetwork requires layout_features.")
        image_embedding = self.image_branch(images)
        layout_embedding = self.layout_branch(layout_features)
        return self.fusion(torch.cat([image_embedding, layout_embedding], dim=1))
