from __future__ import annotations

import re
from html.parser import HTMLParser

from .w3c_models import HTMLNode, W3CDOMNode, W3CSection


class _DOMParser(HTMLParser):

    SKIP_TAGS = {"script", "style", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.root = HTMLNode("document")
        self.current = self.root
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP_TAGS:
            # 2026.05.25, hhlee, 문서 본문만 쓰려고 코드랑 그림은 건너뜀
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        node = HTMLNode(tag=tag, attrs={key: value or "" for key, value in attrs}, parent=self.current)
        self.current.children.append(node)
        if tag not in {"br", "hr", "img", "meta", "link", "input"}:
            self.current = node

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        cursor = self.current
        while cursor.parent is not None:
            if cursor.tag == tag:
                self.current = cursor.parent
                return
            cursor = cursor.parent

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self.current.text_parts.append(text)


class W3CDocumentParser:

    HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    NUMBER_RE = re.compile(r"^(?P<number>(?:[A-Z]\.)?\d+(?:\.\d+)*|[A-Z]\.?)\s+")

    def parse_dom_nodes(self, source_url: str, html: str) -> list[W3CDOMNode]:
        parser = _DOMParser()
        parser.feed(html)
        nodes: list[W3CDOMNode] = []
        heading_context = ""
        heading_level: int | None = None

        for node in self._flatten(parser.root):
            if node.tag == "document":
                continue
            depth = self._depth(node)
            xpath = self._xpath(node)
            parent_xpath = self._xpath(node.parent) if node.parent and node.parent.tag != "document" else ""
            own_text = " ".join(" ".join(node.text_parts).split())
            # 2026.05.25, scyang, 출처 찾기 쉽게 주변 제목도 같이 들고 감
            if node.tag in self.HEADING_TAGS and node.text():
                heading_context = node.text()
                heading_level = int(node.tag[1])
            nodes.append(
                W3CDOMNode(
                    source_url=source_url,
                    tag=node.tag,
                    attrs=dict(node.attrs),
                    text=own_text,
                    depth=depth,
                    xpath=xpath,
                    parent_xpath=parent_xpath,
                    heading_context=heading_context,
                    heading_level=heading_level,
                )
            )
        return nodes

    def parse_sections(self, source_url: str, html: str) -> list[W3CSection]:
        parser = _DOMParser()
        parser.feed(html)
        headings = self._collect_headings(parser.root)
        sections: list[W3CSection] = []
        for index, heading_node in enumerate(headings):
            next_heading = headings[index + 1] if index + 1 < len(headings) else None
            heading_text = heading_node.text()
            heading_level = int(heading_node.tag[1])
            section_text = self._text_until_next_heading(heading_node, next_heading)
            section_number = self._section_number(heading_text)
            sections.append(
                W3CSection(
                    source_url=source_url,
                    heading=heading_text,
                    heading_level=heading_level,
                    heading_id=heading_node.attrs.get("id", ""),
                    xpath=self._xpath(heading_node),
                    section_number=section_number,
                    parent_numbers=self._parent_numbers(section_number),
                    text=section_text,
                )
            )
        return sections

    def _collect_headings(self, root: HTMLNode) -> list[HTMLNode]:
        headings: list[HTMLNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.tag in self.HEADING_TAGS and node.text():
                headings.append(node)
            stack.extend(reversed(node.children))
        return headings

    def _text_until_next_heading(self, heading: HTMLNode, next_heading: HTMLNode | None) -> str:
        root = self._root(heading)
        ordered = self._flatten(root)
        start = ordered.index(heading)
        end = ordered.index(next_heading) if next_heading is not None else len(ordered)
        parts = []
        for node in ordered[start + 1 : end]:
            if node.tag in self.HEADING_TAGS:
                continue
            text = " ".join(node.text_parts)
            if text:
                parts.append(text)
        return " ".join(" ".join(parts).split())

    def _flatten(self, root: HTMLNode) -> list[HTMLNode]:
        ordered: list[HTMLNode] = []

        def visit(node: HTMLNode) -> None:
            ordered.append(node)
            for child in node.children:
                visit(child)

        visit(root)
        return ordered

    def _root(self, node: HTMLNode) -> HTMLNode:
        cursor = node
        while cursor.parent is not None:
            cursor = cursor.parent
        return cursor

    def _xpath(self, node: HTMLNode) -> str:
        if node.tag == "document":
            return ""
        parts: list[str] = []
        cursor: HTMLNode | None = node
        while cursor is not None and cursor.tag != "document":
            if cursor.parent is None:
                break
            same_tag_siblings = [child for child in cursor.parent.children if child.tag == cursor.tag]
            index = next(index for index, sibling in enumerate(same_tag_siblings, start=1) if sibling is cursor)
            node_id = cursor.attrs.get("id")
            suffix = f"[@id='{node_id}']" if node_id else f"[{index}]"
            parts.append(f"{cursor.tag}{suffix}")
            cursor = cursor.parent
        return "/" + "/".join(reversed(parts))

    def _depth(self, node: HTMLNode) -> int:
        depth = 0
        cursor = node.parent
        while cursor is not None and cursor.tag != "document":
            depth += 1
            cursor = cursor.parent
        return depth

    def _section_number(self, heading: str) -> str:
        match = self.NUMBER_RE.match(heading)
        return match.group("number").rstrip(".") if match else ""

    def _parent_numbers(self, section_number: str) -> list[str]:
        if not section_number or "." not in section_number:
            return []
        tokens = section_number.split(".")
        return [".".join(tokens[:index]) for index in range(1, len(tokens))]
