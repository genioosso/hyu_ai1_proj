from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class HTMLNode:

    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["HTMLNode"] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    parent: "HTMLNode | None" = None

    def text(self) -> str:
        parts = list(self.text_parts)
        for child in self.children:
            parts.append(child.text())
        return " ".join(" ".join(parts).split())


@dataclass(frozen=True)
class W3CSourceDocument:

    key: str
    url: str


@dataclass(frozen=True)
class W3CDOMNode:

    source_url: str
    tag: str
    attrs: dict[str, str]
    text: str
    depth: int
    xpath: str
    parent_xpath: str
    heading_context: str
    heading_level: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class W3CSection:

    source_url: str
    heading: str
    heading_level: int
    heading_id: str
    xpath: str
    section_number: str
    parent_numbers: list[str]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class W3CUsabilitySignal:

    signal_id: str
    name: str
    category: str
    source_key: str
    source_url: str
    source_heading: str
    source_xpath: str
    source_section_number: str
    observability: str
    measurable_signals: list[str]
    evidence_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
