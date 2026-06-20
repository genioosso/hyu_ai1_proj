from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from .w3c_models import W3CSection, W3CUsabilitySignal


CATEGORY_RULES: dict[str, dict[str, Any]] = {
    "visual_perceivability": {
        "keywords": ["perceivable", "text alternative", "contrast", "distinguishable", "reflow", "spacing", "color", "non-text"],
        "signals": ["text_count", "image_count", "layout_density", "whitespace_ratio", "component_bounds"],
        "observability": "partial",
        "weight": 0.24,
    },
    "operable_navigation": {
        "keywords": ["operable", "keyboard", "focus", "pointer", "target size", "dragging", "navigation", "bypass"],
        "signals": ["clickable_count", "button_count", "component_density", "touch_target_bounds", "focus_or_action_label_presence"],
        "observability": "partial",
        "weight": 0.24,
    },
    "understandable_labels": {
        "keywords": ["understandable", "readable", "predictable", "label", "instruction", "error", "consistent", "help"],
        "signals": ["text_ratio", "input_count", "labeled_input_presence", "component_type_distribution"],
        "observability": "partial",
        "weight": 0.24,
    },
    "robust_semantics": {
        "keywords": ["robust", "compatible", "name, role, value", "status messages", "programmatically determined", "markup"],
        "signals": ["component_class", "component_label", "text", "clickable", "enabled", "hierarchy_depth"],
        "observability": "partial",
        "weight": 0.18,
    },
    "runtime_or_process_only": {
        "keywords": ["captions", "audio", "video", "timing", "timeout", "authentication", "flashes", "animation", "complete process"],
        "signals": [],
        "observability": "not_directly_observable",
        "weight": 0.10,
    },
}


class W3CSignalClassifier:

    def extract_signals(self, sections: list[W3CSection]) -> list[W3CUsabilitySignal]:
        signals: list[W3CUsabilitySignal] = []
        for section in sections:
            category = self.classify(section)
            if not category:
                continue
            rule = CATEGORY_RULES[category]
            source_key = self.source_key(section.source_url)
            signal_id = self.signal_id(source_key, section)
            signals.append(
                W3CUsabilitySignal(
                    signal_id=signal_id,
                    name=section.heading,
                    category=category,
                    source_key=source_key,
                    source_url=section.source_url,
                    source_heading=section.heading,
                    source_xpath=section.xpath,
                    source_section_number=section.section_number,
                    observability=str(rule["observability"]),
                    measurable_signals=list(rule["signals"]),
                    evidence_summary=self.summary(section.text),
                )
            )
        return self.dedupe_signals(signals)

    def classify(self, section: W3CSection) -> str:
        haystack = f"{section.heading} {section.text}".lower()
        if not self.is_relevant_wcag_section(section):
            return ""
        # 2026.06.02, scyang, 화면에서 못 보는 기준은 여기서 힘 빼고 봄
        scores: dict[str, int] = {}
        for category, rule in CATEGORY_RULES.items():
            scores[category] = sum(1 for keyword in rule["keywords"] if keyword in haystack)
        best_category, best_score = max(scores.items(), key=lambda item: item[1])
        return best_category if best_score > 0 else ""

    def is_relevant_wcag_section(self, section: W3CSection) -> bool:
        heading = section.heading.lower()
        if heading in {"table of contents", "abstract", "status of this document"}:
            return False
        return (
            "success criterion" in heading
            or "guideline" in heading
            or "core requirement" in heading
            or "supplemental requirement" in heading
            or "assertion" in heading
            or "outcome" in heading
            or "principle" in heading
            or heading in {"perceivable 1.", "operable 2.", "understandable 3.", "robust 4."}
        )

    def source_key(self, source_url: str) -> str:
        lowered = source_url.lower()
        if "wcag20" in lowered:
            return "wcag20"
        if "wcag21" in lowered:
            return "wcag21"
        if "wcag22" in lowered:
            return "wcag22"
        if "wcag-3.0" in lowered:
            return "wcag30"
        return "w3c"

    def signal_id(self, source_key: str, section: W3CSection) -> str:
        base = section.section_number or section.heading
        slug = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
        return f"{source_key}_{slug}"

    def summary(self, text: str) -> str:
        return text[:700].strip()

    def dedupe_signals(self, signals: list[W3CUsabilitySignal]) -> list[W3CUsabilitySignal]:
        seen: set[str] = set()
        deduped: list[W3CUsabilitySignal] = []
        for signal in signals:
            # 2026.06.02, scyang, 중복 절은 처음 잡힌 것만 남겨도 충분함
            if signal.signal_id in seen:
                continue
            seen.add(signal.signal_id)
            deduped.append(signal)
        return deduped


class W3CCriteriaBuilder:

    def aggregate(self, signals: list[W3CUsabilitySignal]) -> list[dict[str, Any]]:
        grouped: dict[str, list[W3CUsabilitySignal]] = defaultdict(list)
        for signal in signals:
            grouped[signal.category].append(signal)

        criteria: list[dict[str, Any]] = []
        for category, rule in CATEGORY_RULES.items():
            category_signals = grouped.get(category, [])
            source_counts = Counter(signal.source_key for signal in category_signals)
            criteria.append(
                {
                    "criterion_id": category,
                    "name": self.display_name(category),
                    "weight": rule["weight"],
                    "observability": rule["observability"],
                    "measurable_signals": rule["signals"],
                    "source_signal_count": len(category_signals),
                    "source_coverage": dict(source_counts),
                    "source_signal_ids": [signal.signal_id for signal in category_signals],
                }
            )
        return criteria

    def display_name(self, category: str) -> str:
        return {
            "visual_perceivability": "Visual Perceivability",
            "operable_navigation": "Operable Navigation and Touch Interaction",
            "understandable_labels": "Understandable Labels and Structure",
            "robust_semantics": "Robust Semantic Compatibility",
            "runtime_or_process_only": "Runtime or Process-only Constraints",
        }[category]
