from __future__ import annotations

from .w3c_models import W3CDOMNode


class W3CEvidenceExtractor:

    EVIDENCE_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "section", "p", "li", "table", "tr", "th", "td", "a", "code", "dfn"}
    TEXT_TAGS = {"p", "li", "th", "td", "a", "code", "dfn"}

    def extract(self, dom_nodes: list[W3CDOMNode]) -> list[W3CDOMNode]:
        evidence: list[W3CDOMNode] = []
        for node in dom_nodes:
            if node.tag not in self.EVIDENCE_TAGS:
                continue
            if node.tag in {"section", "table", "tr"} and not node.attrs.get("id") and not node.text:
                continue
            if node.tag in self.TEXT_TAGS and not node.text:
                continue
            if node.tag in self.TEXT_TAGS and len(node.text.strip()) < 12:
                continue
            if node.tag == "a" and not node.attrs.get("href"):
                continue
            evidence.append(node)
        return evidence
