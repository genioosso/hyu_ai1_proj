from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datasets.dataset_manager import DatasetManager
from datasets.rico_parser import RicoParser
from evaluators.evaluator import Evaluator
from experiments.w3c_rule_flow.config import W3CRuleExperimentConfig
from features.data_preprocessor import DataPreprocessor
from features.layout_feature_extractor import LayoutFeatureExtractor
from models.model_factory import ModelFactory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "w3c_rule_cv_experiment_full.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    config = W3CRuleExperimentConfig.from_yaml(raw_config, args.config, output_dir=args.output_dir)
    labels_path = config.dataset_root / "w3c_rule_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Label manifest not found: {labels_path}")

    parser_obj = RicoParser()
    layout_extractor = LayoutFeatureExtractor()
    dataset_manager = DatasetManager(config.dataset_root, labels_path, parser_obj, layout_extractor, seed=config.seed)
    samples = dataset_manager.load_samples()
    split = dataset_manager.split(samples, config.train_ratio, config.validation_ratio)
    preprocessor = DataPreprocessor(config.image_size)
    loader = dataset_manager.create_loader(
        split.test,
        preprocessor.build_transform(False),
        config.batch_size,
        False,
        num_workers=config.num_workers,
        cache_dir=config.cache_dir if config.cache_enabled else None,
        cache_namespace=f"image{config.image_size}",
    )

    device = torch.device(config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu")
    factory = ModelFactory(num_classes=2, layout_feature_dim=layout_extractor.feature_dim)
    evaluator = Evaluator(device)
    rows_by_model: dict[str, list[dict[str, object]]] = {}

    for model_name in config.model_names:
        checkpoint_path = config.output_dir / "checkpoints" / model_name / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = factory.create(model_name).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        result = evaluator.evaluate(model, loader)
        rows_by_model[model_name] = result.prediction_rows
        write_model_predictions(config.output_dir, model_name, result.prediction_rows)
        print(f"exported {model_name}: {len(result.prediction_rows)} rows")

    write_combined_predictions(config.output_dir, rows_by_model)
    print(config.output_dir / "all_models_test_predictions.csv")


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
    with (output_dir / "all_models_test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id in sorted(combined, key=lambda value: int(value) if value.isdigit() else value):
            writer.writerow(combined[sample_id])


if __name__ == "__main__":
    main()
