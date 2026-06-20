from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.w3c_rule_flow import W3CRuleCVExperiment, W3CRuleExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "w3c_rule_cv_experiment.yaml")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = build_experiment_from_config(
        config=config,
        config_path=args.config,
        sample_limit=args.sample_limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        model_names=args.models,
        seed=args.seed,
        output_dir=args.output_dir,
    ).run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_experiment_from_config(
    config: dict[str, Any],
    config_path: Path,
    sample_limit: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    image_size: int | None = None,
    model_names: list[str] | None = None,
    seed: int | None = None,
    output_dir: Path | None = None,
) -> W3CRuleCVExperiment:
    experiment_config = W3CRuleExperimentConfig.from_yaml(
        config=config,
        config_path=config_path,
        sample_limit=sample_limit,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        model_names=model_names,
        seed=seed,
        output_dir=output_dir,
    )
    return W3CRuleCVExperiment(experiment_config)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()

