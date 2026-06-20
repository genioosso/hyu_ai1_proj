from __future__ import annotations

import argparse
import json
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent

from experiments.build_w3c_usability_spec import W3CUsabilitySpecBuilder
from experiments.run_w3c_rule_cv_experiment import build_experiment_from_config, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["w3c-cv", "w3c-spec"],
        default="w3c-cv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "w3c_rule_cv_experiment_full.yaml",
    )
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--refresh-w3c", action="store_true")
    args = parser.parse_args()

    if args.mode == "w3c-spec":
        artifacts = W3CUsabilitySpecBuilder(PROJECT_ROOT).build(refresh=args.refresh_w3c, allow_network=args.refresh_w3c)
        print(json.dumps({name: path.relative_to(PROJECT_ROOT).as_posix() for name, path in artifacts.items()}, indent=2))
        return

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


if __name__ == "__main__":
    main()
