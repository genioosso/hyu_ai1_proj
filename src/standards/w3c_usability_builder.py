from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .w3c_artifact_writer import W3CArtifactWriter
from .w3c_document_parser import W3CDocumentParser
from .w3c_evidence_extractor import W3CEvidenceExtractor
from .w3c_fetcher import W3CFetcher
from .w3c_models import W3CSourceDocument, W3CDOMNode, W3CSection
from .w3c_signal_classifier import W3CCriteriaBuilder, W3CSignalClassifier


class W3CUsabilitySpecBuilder:

    DEFAULT_SOURCES = (
        W3CSourceDocument("wcag20", "https://www.w3.org/TR/WCAG20/"),
        W3CSourceDocument("wcag21", "https://www.w3.org/TR/WCAG21/"),
        W3CSourceDocument("wcag22", "https://www.w3.org/TR/WCAG22/"),
        W3CSourceDocument("wcag30", "https://www.w3.org/TR/wcag-3.0/"),
    )

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.output_dir = project_root / "outputs" / "standards" / "w3c_usability"
        self.cache_dir = self.output_dir / "cache"
        self.raw_dom_dir = self.output_dir / "raw_dom"
        self.extracted_dir = self.output_dir / "extracted"
        self.sections_path = self.extracted_dir / "w3c_sections.json"
        self.evidence_nodes_path = self.extracted_dir / "w3c_evidence_nodes.json"
        self.signals_path = self.output_dir / "w3c_usability_signals.json"
        self.criteria_path = self.output_dir / "w3c_usability_criteria.json"
        self.traceability_path = self.output_dir / "w3c_traceability_matrix.json"
        self.rubric_path = project_root / "configs" / "w3c_usability_rubric.yaml"
        self.parser = W3CDocumentParser()
        self.fetcher = W3CFetcher(self.cache_dir)
        self.evidence_extractor = W3CEvidenceExtractor()
        self.signal_classifier = W3CSignalClassifier()
        self.criteria_builder = W3CCriteriaBuilder()
        self.writer = W3CArtifactWriter()

    def build(self, refresh: bool = False, timeout_seconds: int = 30, allow_network: bool = False) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dom_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        all_sections: list[W3CSection] = []
        all_evidence_nodes: list[W3CDOMNode] = []
        for source in self.DEFAULT_SOURCES:
            html = self.fetcher.fetch_or_cache(
                source,
                refresh=refresh,
                timeout_seconds=timeout_seconds,
                allow_network=allow_network,
            )
            dom_nodes = self.parser.parse_dom_nodes(source.url, html)
            sections = self.parser.parse_sections(source.url, html)
            self.writer.write_json(
                self.raw_dom_dir / f"{source.key}_dom.json",
                {
                    "source": asdict(source),
                    "node_count": len(dom_nodes),
                    "nodes": [node.to_dict() for node in dom_nodes],
                },
            )
            all_sections.extend(sections)
            all_evidence_nodes.extend(self.evidence_extractor.extract(dom_nodes))

        signals = self.signal_classifier.extract_signals(all_sections)
        criteria = self.criteria_builder.aggregate(signals)
        self.writer.write_json(
            self.sections_path,
            {
                "sources": [asdict(source) for source in self.DEFAULT_SOURCES],
                "section_count": len(all_sections),
                "sections": [section.to_dict() for section in all_sections],
            },
        )
        self.writer.write_json(
            self.evidence_nodes_path,
            {
                "definition": "Filtered W3C DOM evidence nodes used as candidate material for usability criteria extraction.",
                "node_count": len(all_evidence_nodes),
                "nodes": [node.to_dict() for node in all_evidence_nodes],
            },
        )
        self.writer.write_json(
            self.signals_path,
            {
                "definition": "XPath-like W3C document sections mapped to Rico-observable usability signals.",
                "signal_count": len(signals),
                "signals": [signal.to_dict() for signal in signals],
            },
        )
        self.writer.write_json(
            self.criteria_path,
            {
                "definition": (
                    "W3C WCAG 2.0, WCAG 2.1, WCAG 2.2, and WCAG 3.0-derived static mobile UI usability proxy. "
                    "Built from crawled W3C source documents and limited to static Rico screen evidence."
                ),
                "limitation": (
                    "The criteria are accessibility-informed weak usability proxies for Rico screenshot/hierarchy data. "
                    "They are not WCAG conformance claims and not user-study usability measurements."
                ),
                "criteria": criteria,
            },
        )
        self.writer.write_json(
            self.traceability_path,
            {
                "traceability": [
                    {
                        "signal_id": signal.signal_id,
                        "category": signal.category,
                        "source_key": signal.source_key,
                        "source_url": signal.source_url,
                        "source_heading": signal.source_heading,
                        "source_xpath": signal.source_xpath,
                        "source_section_number": signal.source_section_number,
                        "observability": signal.observability,
                    }
                    for signal in signals
                ]
            },
        )
        self.writer.write_rubric(self.rubric_path, self.DEFAULT_SOURCES, criteria)
        return {
            "sections": self.sections_path,
            "evidence_nodes": self.evidence_nodes_path,
            "signals": self.signals_path,
            "criteria": self.criteria_path,
            "traceability_matrix": self.traceability_path,
            "rubric": self.rubric_path,
            "cache_dir": self.cache_dir,
            "raw_dom_dir": self.raw_dom_dir,
        }
