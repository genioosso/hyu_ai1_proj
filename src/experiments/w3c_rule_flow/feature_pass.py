from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets.rico_parser import RicoParser
from features.layout_feature_extractor import LayoutFeatureExtractor
from features.sample_feature_extractor import SampleFeatureExtractor

from .sample_picker import RicoSamplePicker


def extract_feature_records(
    sample_ids: list[str],
    dataset_root: Path,
    output_dir: Path,
    sample_picker: RicoSamplePicker,
    parser: RicoParser,
    layout_extractor: LayoutFeatureExtractor,
    display_path: Callable[[Path | str], str],
) -> list[dict[str, Any]]:
    output_path = output_dir / "dataset_features.jsonl"
    # 2026.06.03, hhlee, 캐시는 샘플 순서까지 같을 때만 믿음
    cached = load_cached_feature_records(output_path, sample_ids)
    if cached:
        print(f"feature extraction: reused {len(cached)} cached records", flush=True)
        return cached

    extractor = SampleFeatureExtractor(parser, layout_extractor)
    records: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()

    with output_path.open("w", encoding="utf-8") as handle:
        for index, sample_id in enumerate(sample_ids, start=1):
            screenshot_path = sample_picker.screenshot_path(sample_id)
            hierarchy_path = dataset_root / "hierarchies" / f"{sample_id}.json"
            record = extractor.extract(sample_id, screenshot_path, hierarchy_path).to_dict()
            record["screenshot_path"] = display_path(record["screenshot_path"])
            record["hierarchy_path"] = display_path(record["hierarchy_path"])
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 2026.06.03, hhlee, 추출 오래 걸려서 진행률 없으면 답답함
            if index == 1 or index % 1000 == 0 or index == len(sample_ids):
                handle.flush()
                elapsed = time.monotonic() - started_at
                print(f"feature extraction: {index}/{len(sample_ids)} samples ({elapsed:.1f}s)", flush=True)

    return records


def load_cached_feature_records(output_path: Path, sample_ids: list[str]) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []

    records: list[dict[str, Any]] = []
    try:
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return []

    if len(records) != len(sample_ids):
        return []
    if [str(record.get("sample_id")) for record in records] != sample_ids:
        return []
    return records
