from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import torch

from datasets.dataset_manager import DatasetManager, DatasetSplit
from datasets.rico_parser import RicoParser
from datasets.ui_dataset import UISample
from evaluators.evaluator import Evaluator
from features.data_preprocessor import DataPreprocessor
from features.layout_feature_extractor import LayoutFeatureExtractor
from models.model_factory import build_model
from trainers.trainer import Trainer
from visualizers.result_plotter import ResultPlotter

from .config import W3CRuleExperimentConfig


def train_models(
    labels_path: Path,
    config: W3CRuleExperimentConfig,
    parser: RicoParser,
    layout_extractor: LayoutFeatureExtractor,
    display_path: Callable[[Path | str], str],
) -> dict[str, dict[str, float]]:
    dataset_manager = DatasetManager(
        config.dataset_root,
        labels_path,
        parser,
        layout_extractor,
        seed=config.seed,
    )
    samples = dataset_manager.load_samples()
    # 2026.06.13, scyang, 라벨 바뀌어도 같은 기준으로 나눠야 비교됨
    split = dataset_manager.split(samples, config.train_ratio, config.validation_ratio)
    write_split_metadata(config, split, display_path)

    preprocessor = DataPreprocessor(image_size=config.image_size)
    cache_dir = config.cache_dir if config.cache_enabled else None
    # 2026.06.13, scyang, 이미지 크기 다르면 캐시도 따로 써야 함
    cache_namespace = f"image{config.image_size}"
    train_loader = dataset_manager.create_loader(
        split.train,
        preprocessor.build_transform(True),
        config.batch_size,
        True,
        num_workers=config.num_workers,
        cache_dir=cache_dir,
        cache_namespace=cache_namespace,
    )
    validation_loader = dataset_manager.create_loader(
        split.validation,
        preprocessor.build_transform(False),
        config.batch_size,
        False,
        num_workers=config.num_workers,
        cache_dir=cache_dir,
        cache_namespace=cache_namespace,
    )
    test_loader = dataset_manager.create_loader(
        split.test,
        preprocessor.build_transform(False),
        config.batch_size,
        False,
        num_workers=config.num_workers,
        cache_dir=cache_dir,
        cache_namespace=cache_namespace,
    )

    device = torch.device(config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(device)
    plotter = ResultPlotter(config.output_dir / "plots")
    metrics: dict[str, dict[str, float]] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    prediction_rows_by_model: dict[str, list[dict[str, object]]] = {}
    # 2026.06.13, scyang, 가중치는 평가 데이터 보지 말고 학습 데이터로만 잡음
    weights = class_weights(split.train).to(device)

    for model_name in config.model_names:
        print(f"training started: {model_name}", flush=True)
        model = build_model(model_name, num_classes=2, layout_feature_dim=layout_extractor.feature_dim)
        trainer = Trainer(
            model,
            device,
            learning_rate=config.learning_rate,
            checkpoint_dir=config.output_dir / "checkpoints" / model_name,
            class_weights=weights,
        )
        history = trainer.train(train_loader, validation_loader, config.epochs)
        # 2026.06.15, scyang, 테스트는 제일 잘 저장된 모델로만 돌림
        trainer.load_checkpoint(config.output_dir / "checkpoints" / model_name / "best.pt")
        histories[model_name] = {
            "train_loss": history.train_loss,
            "train_accuracy": history.train_accuracy,
            "validation_loss": history.validation_loss,
            "validation_accuracy": history.validation_accuracy,
            "validation_precision": history.validation_precision,
            "validation_recall": history.validation_recall,
        }
        result = evaluator.evaluate(model, test_loader)
        plotter.plot_history(model_name, history)
        plotter.plot_confusion_matrix(model_name, result)
        prediction_rows_by_model[model_name] = result.prediction_rows
        write_model_predictions(config.output_dir, model_name, result.prediction_rows)
        metrics[model_name] = {
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "roc_auc": result.roc_auc,
        }
        print(f"training finished: {model_name} metrics={metrics[model_name]}", flush=True)

    (config.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (config.output_dir / "training_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    write_combined_predictions(config.output_dir, prediction_rows_by_model)
    return metrics


def class_weights(train_samples: list[UISample]) -> torch.Tensor:
    counts = [0, 0]
    for sample in train_samples:
        if sample.label in (0, 1):
            counts[sample.label] += 1
    total = max(sum(counts), 1)
    return torch.tensor(
        [total / max(counts[0], 1), total / max(counts[1], 1)],
        dtype=torch.float32,
    )


def write_split_metadata(
    config: W3CRuleExperimentConfig,
    split: DatasetSplit,
    display_path: Callable[[Path | str], str],
) -> None:
    payload = {
        "train_count": len(split.train),
        "validation_count": len(split.validation),
        "test_count": len(split.test),
        "total_count": len(split.train) + len(split.validation) + len(split.test),
        "train_ratio": config.train_ratio,
        "validation_ratio": config.validation_ratio,
        "test_ratio": max(0.0, 1.0 - config.train_ratio - config.validation_ratio),
        "seed": config.seed,
        "sample_limit": config.sample_limit,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "image_size": config.image_size,
        "models": config.model_names,
        "cache_enabled": config.cache_enabled,
        "cache_dir": display_path(config.cache_dir),
        "num_workers": config.num_workers,
        "class_weighting": "inverse_frequency_from_train_split",
        "test_model_selection": "best validation checkpoint",
    }
    (config.output_dir / "split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_model_predictions(output_dir: Path, model_name: str, rows: list[dict[str, object]]) -> None:
    output_path = output_dir / f"{model_name}_test_predictions.csv"
    fieldnames = [
        "sample_id",
        "true_label",
        "true_class",
        "predicted_label",
        "predicted_class",
        "usable_probability",
        "is_correct",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_combined_predictions(output_dir: Path, rows_by_model: dict[str, list[dict[str, object]]]) -> None:
    if not rows_by_model:
        return
    # 2026.06.17, scyang, 예측 파일 합쳐두면 모델별 실수 찾기 쉬움
    combined: dict[str, dict[str, object]] = {}
    for model_name, rows in rows_by_model.items():
        for row in rows:
            sample_id = str(row["sample_id"])
            entry = combined.setdefault(
                sample_id,
                {
                    "sample_id": sample_id,
                    "true_label": row["true_label"],
                    "true_class": row["true_class"],
                },
            )
            entry[f"{model_name}_predicted_label"] = row["predicted_label"]
            entry[f"{model_name}_predicted_class"] = row["predicted_class"]
            entry[f"{model_name}_usable_probability"] = row["usable_probability"]
            entry[f"{model_name}_is_correct"] = row["is_correct"]

    model_names = list(rows_by_model.keys())
    fieldnames = ["sample_id", "true_label", "true_class"]
    for model_name in model_names:
        fieldnames.extend(
            [
                f"{model_name}_predicted_label",
                f"{model_name}_predicted_class",
                f"{model_name}_usable_probability",
                f"{model_name}_is_correct",
            ]
        )
    output_path = output_dir / "all_models_test_predictions.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id in sorted(combined, key=lambda value: int(value) if value.isdigit() else value):
            writer.writerow(combined[sample_id])
