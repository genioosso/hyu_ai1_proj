from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UIComponent:

    component_type: str
    bounds: tuple[int, int, int, int]
    text: str = ""
    clickable: bool = False
    enabled: bool = True


class RicoParser:

    TYPE_ALIASES = {
        "button": "button",
        "imagebutton": "button",
        "textview": "text",
        "edittext": "input",
        "imageview": "image",
    }

    def parse_file(self, hierarchy_path: Path) -> list[UIComponent]:
        with hierarchy_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return self.parse_payload(payload)

    def parse_payload(self, payload: dict[str, Any]) -> list[UIComponent]:
        components: list[UIComponent] = []
        root = payload.get("activity", {}).get("root") or payload.get("root") or payload
        self._walk(root, components)
        return components

    def _walk(self, node: dict[str, Any], components: list[UIComponent]) -> None:
        raw_type = str(node.get("class") or node.get("className") or node.get("componentLabel") or "unknown")
        component_type = self._normalize_type(raw_type)
        bounds = self._parse_bounds(node.get("bounds") or node.get("visibleBounds") or [0, 0, 0, 0])
        components.append(
            UIComponent(
                component_type=component_type,
                bounds=bounds,
                text=str(node.get("text") or ""),
                clickable=bool(node.get("clickable", False)),
                enabled=bool(node.get("enabled", True)),
            )
        )
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                self._walk(child, components)

    def _normalize_type(self, raw_type: str) -> str:
        lowered = raw_type.lower().split(".")[-1]
        for token, normalized in self.TYPE_ALIASES.items():
            if token in lowered:
                return normalized
        return "container" if "layout" in lowered or "viewgroup" in lowered else "other"

    def _parse_bounds(self, raw_bounds: Any) -> tuple[int, int, int, int]:
        if isinstance(raw_bounds, list) and len(raw_bounds) == 4:
            if all(isinstance(item, list) for item in raw_bounds[:2]):
                x1, y1 = raw_bounds[0]
                x2, y2 = raw_bounds[1]
                return int(x1), int(y1), int(x2), int(y2)
            x1, y1, x2, y2 = (int(value) for value in raw_bounds)
            return x1, y1, x2, y2
        return 0, 0, 0, 0
