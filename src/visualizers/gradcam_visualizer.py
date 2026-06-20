from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from models.base_model import BaseModel


class GradCAMVisualizer:

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        model: BaseModel,
        image: torch.Tensor,
        layout: torch.Tensor,
        output_name: str,
        target_class: int | None = None,
    ) -> Path:
        model.eval()
        target_layer = self._last_conv2d(model)
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        # 2026.06.13, hhlee, 마지막 합성곱 층만 보면 근거 위치 대충 보임
        def forward_hook(_module, _inputs, output):
            activations.append(output)
            output.register_hook(lambda grad: gradients.append(grad))

        handle = target_layer.register_forward_hook(forward_hook)
        try:
            model.zero_grad(set_to_none=True)
            logits = model(image.unsqueeze(0), layout.unsqueeze(0))
            target = int(torch.argmax(logits, dim=1).item()) if target_class is None else target_class
            logits[0, target].backward()
        finally:
            handle.remove()

        cam = self._build_cam(activations[-1].detach()[0], gradients[-1].detach()[0])
        path = self.output_dir / f"{output_name}_gradcam.png"
        self._save_overlay(image, cam, path)
        return path

    def _last_conv2d(self, model: nn.Module) -> nn.Conv2d:
        layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        if not layers:
            raise ValueError(f"No Conv2d layer found in {model.__class__.__name__}")
        return layers[-1]

    def _build_cam(self, activation: torch.Tensor, gradient: torch.Tensor) -> np.ndarray:
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=0)).cpu().numpy()
        cam -= cam.min()
        max_value = cam.max()
        return cam / max_value if max_value > 0 else cam

    def _save_overlay(self, image: torch.Tensor, cam: np.ndarray, path: Path) -> None:
        base = self._denormalize(image).permute(1, 2, 0).cpu().numpy()
        base = np.clip(base, 0.0, 1.0)
        cam_image = Image.fromarray(np.uint8(cam * 255)).resize((base.shape[1], base.shape[0]), Image.Resampling.BILINEAR)
        heat = plt.get_cmap("jet")(np.asarray(cam_image, dtype=np.float32) / 255.0)[..., :3]
        # 2026.06.13, hhlee, 열지도가 너무 진하면 원본 화면 안 보여서 낮춤
        overlay = np.clip(base * 0.55 + heat * 0.45, 0.0, 1.0)

        fig, axis = plt.subplots(figsize=(4, 4))
        axis.imshow(overlay)
        axis.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(path)
        plt.close(fig)

    def _denormalize(self, image: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=image.dtype, device=image.device).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=image.dtype, device=image.device).view(3, 1, 1)
        return image * std + mean
