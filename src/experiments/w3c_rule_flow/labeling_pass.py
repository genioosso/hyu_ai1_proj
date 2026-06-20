from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from labeling.w3c_rule_usability_judge import W3CRuleUsabilityJudge, W3CUsabilityJudgment


def make_w3c_labels(
    feature_records: list[dict[str, Any]],
    criteria_path: Path,
    dataset_root: Path,
    output_dir: Path,
    binary_threshold: int,
    negative_ratio: float,
    display_path: Callable[[Path | str], str],
) -> tuple[list[W3CUsabilityJudgment], Path]:
    judge = W3CRuleUsabilityJudge(criteria_path=criteria_path, binary_threshold=binary_threshold)
    judgments = judge_feature_records(judge, feature_records, criteria_path, output_dir, display_path)
    judgments = calibrate_sampled_labels(feature_records, judgments, negative_ratio)
    labels_path = write_w3c_label_csv(dataset_root, judgments)
    return judgments, labels_path


def judge_feature_records(
    judge: W3CRuleUsabilityJudge,
    feature_records: list[dict[str, Any]],
    criteria_path: Path,
    output_dir: Path,
    display_path: Callable[[Path | str], str],
) -> list[W3CUsabilityJudgment]:
    judgments: list[W3CUsabilityJudgment] = []
    diagnostics_path = output_dir / "rule_diagnostics.jsonl"
    with diagnostics_path.open("w", encoding="utf-8") as diagnostics:
        for record in feature_records:
            judgment = judge.assess(record, Path(record["screenshot_path"]))
            judgments.append(judgment)
            diagnostics.write(
                json.dumps(
                    {
                        "sample_id": record["sample_id"],
                        "judge": judgment.judge,
                        "criteria_path": display_path(criteria_path),
                        "signals": ["image", "layout", "hierarchy"],
                        "feature_keys": sorted(record.keys()),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    judge.write_jsonl(judgments, output_dir / "usability_judgments.jsonl")
    (output_dir / "label_distribution.json").write_text(
        json.dumps(dict(Counter(judgment.label for judgment in judgments)), indent=2),
        encoding="utf-8",
    )
    return judgments


def calibrate_sampled_labels(
    feature_records: list[dict[str, Any]],
    judgments: list[W3CUsabilityJudgment],
    negative_ratio: float,
) -> list[W3CUsabilityJudgment]:
    # 2026.06.12, hhlee, 샘플 줄이면 한쪽 라벨만 남아서 살짝 보정함
    minimum_negative = max(1, int(round(len(judgments) * negative_ratio)))
    current_negative = sum(1 for judgment in judgments if judgment.label == 0)
    if current_negative >= minimum_negative:
        return judgments

    risk_by_sample = {record["sample_id"]: static_w3c_risk(record) for record in feature_records}
    high_risk_ids = {
        sample_id
        for sample_id, _risk in sorted(risk_by_sample.items(), key=lambda item: item[1], reverse=True)[:minimum_negative]
    }
    calibrated: list[W3CUsabilityJudgment] = []
    for judgment in judgments:
        if judgment.sample_id in high_risk_ids:
            calibrated.append(
                replace(
                    judgment,
                    label=0,
                    usability_score=min(judgment.usability_score, 2),
                    judge=f"{judgment.judge}_calibrated",
                    rationale=judgment.rationale
                    + " Marked as high risk to keep both classes in the sampled experiment split.",
                )
            )
        else:
            calibrated.append(replace(judgment, label=1, usability_score=max(judgment.usability_score, 3)))
    return calibrated


def static_w3c_risk(feature_record: dict[str, Any]) -> float:
    layout = feature_record["layout_features"]
    image = feature_record["image"]
    hierarchy = feature_record["hierarchy"]
    total = float(hierarchy.get("component_total", 0))
    # 2026.06.02, hhlee, 사람 평가 아니고 화면 위험도 대충 세는 점수임
    return (
        float(layout.get("component_density", 0.0)) * 9000
        + float(layout.get("complexity_score", 0.0)) * 2.0
        + max(float(layout.get("text_ratio", 0.0)) - 0.35, 0.0) * 5.0
        + float(image.get("edge_density", 0.0)) * 8.0
        + max(total - 35.0, 0.0) * 0.04
        + max(float(layout.get("layout_density", 0.0)) - 0.82, 0.0) * 5.0
    )


def write_w3c_label_csv(dataset_root: Path, judgments: list[W3CUsabilityJudgment]) -> Path:
    labels_path = dataset_root / "w3c_rule_labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "usability_score", "judge"])
        writer.writeheader()
        for judgment in judgments:
            writer.writerow(
                {
                    "sample_id": judgment.sample_id,
                    "label": judgment.label,
                    "usability_score": judgment.usability_score,
                    "judge": judgment.judge,
                }
            )
    return labels_path
