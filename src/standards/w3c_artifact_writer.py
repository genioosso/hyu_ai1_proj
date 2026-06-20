from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .w3c_models import W3CSourceDocument


class W3CArtifactWriter:

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_rubric(self, path: Path, sources: tuple[W3CSourceDocument, ...], criteria: list[dict[str, Any]]) -> None:
        payload = {
            "version": 1,
            "source": {
                "standards": [source.key for source in sources],
                "urls": [source.url for source in sources],
                "builder": "standards.W3CUsabilitySpecBuilder",
            },
            "labeling": {
                "default_strategy": "w3c_static_ui_heuristic",
                "binary_threshold": 3,
                "output_dir": "outputs/labels/weak_labels",
            },
            "criteria": {
                criterion["criterion_id"]: {
                    "weight": criterion["weight"],
                    "name": criterion["name"],
                    "observability": criterion["observability"],
                    "measurable_signals": criterion["measurable_signals"],
                    "source_signal_count": criterion["source_signal_count"],
                    "source_coverage": criterion["source_coverage"],
                }
                for criterion in criteria
            },
            "heuristic_thresholds": {
                "high_density": 0.00008,
                "high_complexity": 1.3,
                "low_whitespace": 0.25,
                "excessive_text_ratio": 0.45,
                "minimum_score": 1,
                "maximum_score": 5,
            },
            "disclosure": (
                "This rubric is built by crawling W3C WCAG source documents and mapping observable sections "
                "to Rico static screenshot/hierarchy signals. It is a weak usability proxy, not a WCAG conformance test."
            ),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
