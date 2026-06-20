from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from standards.w3c_usability_builder import W3CUsabilitySpecBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    artifacts = W3CUsabilitySpecBuilder(ROOT).build(
        refresh=args.refresh,
        timeout_seconds=args.timeout,
        allow_network=args.refresh,
    )
    print(json.dumps({name: path.relative_to(ROOT).as_posix() for name, path in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
