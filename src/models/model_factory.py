from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .base_model import BaseModel
from .cnn_model import CNNModel
from .efficientnet_model import EfficientNetModel
from .resnet_model import ResNetModel
from .ui_aware_hybrid_network import UIAwareHybridNetwork


def build_model(model_name: str, num_classes: int, layout_feature_dim: int) -> BaseModel:
    normalized = model_name.lower().replace("-", "_")
    builders: dict[str, Callable[[], BaseModel]] = {
        "cnn": lambda: CNNModel(num_classes),
        "resnet18": lambda: ResNetModel(num_classes),
        "efficientnet_b0": lambda: EfficientNetModel(num_classes),
        "ui_aware_hybrid": lambda: UIAwareHybridNetwork(num_classes, layout_feature_dim),
        "uiawarehybridnetwork": lambda: UIAwareHybridNetwork(num_classes, layout_feature_dim),
    }
    try:
        return builders[normalized]()
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc


@dataclass(frozen=True)
class ModelFactory:

    num_classes: int
    layout_feature_dim: int

    def create(self, model_name: str) -> BaseModel:
        return build_model(model_name, self.num_classes, self.layout_feature_dim)
