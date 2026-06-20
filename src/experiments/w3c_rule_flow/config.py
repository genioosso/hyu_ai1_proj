from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class W3CRuleExperimentConfig:

    project_root: Path
    dataset_root: Path
    output_dir: Path
    cache_dir: Path
    sample_limit: int
    epochs: int
    batch_size: int
    image_size: int
    train_ratio: float
    validation_ratio: float
    binary_threshold: int
    negative_ratio: float
    learning_rate: float
    device: str
    num_workers: int
    cache_enabled: bool
    model_names: list[str]
    seed: int

    @classmethod
    def from_yaml(
        cls,
        config: dict[str, Any],
        config_path: Path,
        sample_limit: int | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        image_size: int | None = None,
        model_names: list[str] | None = None,
        seed: int | None = None,
        output_dir: Path | None = None,
    ) -> "W3CRuleExperimentConfig":
        project_root = config_path.resolve().parents[1]
        configured_output = _optional_path(config.get("experiment", {}).get("output_dir"))
        configured_cache = _optional_path(config.get("cache", {}).get("dir")) or Path("outputs/cache/tensors")
        return cls(
            project_root=project_root,
            dataset_root=project_root / "data",
            output_dir=_resolve(project_root, output_dir or configured_output or Path("outputs/runs/w3c_rule_cv_smoke")),
            cache_dir=_resolve(project_root, configured_cache),
            sample_limit=sample_limit if sample_limit is not None else int(config["dataset"]["sample_limit"]),
            epochs=epochs if epochs is not None else int(config["training"]["epochs"]),
            batch_size=batch_size if batch_size is not None else int(config["training"]["batch_size"]),
            image_size=image_size if image_size is not None else int(config["dataset"]["image_size"]),
            train_ratio=float(config["dataset"]["train_ratio"]),
            validation_ratio=float(config["dataset"]["validation_ratio"]),
            binary_threshold=int(config["labeling"].get("binary_threshold", 3)),
            negative_ratio=float(config["labeling"].get("negative_calibration_ratio", 0.25)),
            learning_rate=float(config["training"].get("learning_rate", 0.001)),
            device=str(config["training"].get("device", "cpu")),
            num_workers=int(config["training"].get("num_workers", 0)),
            cache_enabled=bool(config.get("cache", {}).get("enabled", False)),
            model_names=model_names if model_names is not None else list(config["experiment"]["models"]),
            seed=seed if seed is not None else int(config["experiment"]["seed"]),
        )


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _resolve(project_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return project_root / path
