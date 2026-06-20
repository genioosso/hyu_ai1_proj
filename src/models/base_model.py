from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModel(nn.Module, ABC):

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, images: torch.Tensor, layout_features: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
