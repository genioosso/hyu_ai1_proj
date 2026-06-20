from __future__ import annotations

from pathlib import Path


class RicoSamplePicker:

    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root

    def list_ids(self, sample_limit: int) -> list[str]:
        screenshots = {
            path.stem
            for path in (self.dataset_root / "screenshots").glob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        }
        hierarchies = {path.stem for path in (self.dataset_root / "hierarchies").glob("*.json")}
        sample_ids = sorted(screenshots & hierarchies, key=lambda value: int(value) if value.isdigit() else value)
        if sample_limit <= 0 or sample_limit >= len(sample_ids):
            return sample_ids
        step = max(len(sample_ids) // sample_limit, 1)
        return sample_ids[::step][:sample_limit]

    def screenshot_path(self, sample_id: str) -> Path:
        for suffix in (".png", ".jpg", ".jpeg"):
            path = self.dataset_root / "screenshots" / f"{sample_id}{suffix}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Screenshot not found for sample_id={sample_id}")
