"""Heading- and clause-aware chunking with citation metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .parsing import ParsedDocument

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
CLAUSE_RE = re.compile(r"^\*{0,2}((?:\d+\.)+\d+|Q\d+)[.)]?\s")
SECTION_NUM_RE = re.compile(r"^((?:\d+\.)*\d+|Q\d+)[.)]?\s")


@dataclass
class Chunk:
    text: str  # includes a breadcrumb line for better retrieval
    section: str  # "3.5", "Q2", "3.1-3.3" or ""
    heading: str
    index: int
    page: int | None = None


def _pack(paragraphs: list[str], max_chars: int, overlap: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for para in paragraphs:
        if current and size + len(para) > max_chars:
            groups.append(current)
            tail = current[-1] if overlap and len(current[-1]) <= overlap else None
            current = [tail] if tail else []
            size = len(tail) if tail else 0
        if len(para) > max_chars:  # hard-split very long paragraphs
            for i in range(0, len(para), max_chars - overlap):
                piece = para[i : i + max_chars]
                if current:
                    groups.append(current)
                    current, size = [], 0
                groups.append([piece])
            continue
        current.append(para)
        size += len(para)
    if current:
        groups.append(current)
    return groups


def _section_label(paragraphs: list[str], heading: str) -> str:
    clauses = [m.group(1) for p in paragraphs if (m := CLAUSE_RE.match(p))]
    if clauses:
        return clauses[0] if len(clauses) == 1 else f"{clauses[0]}-{clauses[-1]}"
    m = SECTION_NUM_RE.match(heading)
    return m.group(1) if m else ""


def _first_sentence(paragraph: str) -> str:
    text = CLAUSE_RE.sub("", paragraph.strip(), count=1)
    label = re.match(r"\*\*(.+?)\*\*[:.]?\s*", text)
    prefix = ""
    if label:  # "**Carry forward.** Up to 10 days ..." -> "Carry forward: Up to 10 days ..."
        prefix = label.group(1).rstrip(".:") + ": "
        text = text[label.end():]
    text = text.replace("**", "")
    match = re.match(r"(.+?[.!?])(\s|$)", text, re.DOTALL)
    return (prefix + (match.group(1) if match else text)).strip()


def _overview_chunk(title: str, sections: list[tuple[str, list[str]]], index: int) -> Chunk:
    """A table-of-contents style chunk that helps broad "what is the X policy" queries."""
    lines = [f"{title} - overview of all sections"]
    for heading, paras in sections:
        if not heading or not paras:
            continue
        lines.append(f"- {heading}: {_first_sentence(paras[0])[:200]}")
    return Chunk(text="\n".join(lines), section="overview", heading="Overview", index=index)


def chunk_document(doc: ParsedDocument, title: str, max_chars: int = 900, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    if doc.pages and not doc.is_markdown:
        sources = [(i + 1, page) for i, page in enumerate(doc.pages) if page.strip()]
    else:
        sources = [(None, doc.text)]

    for page_no, text in sources:
        sections: list[tuple[str, list[str]]] = []
        heading = ""
        paragraphs: list[str] = []
        for block in re.split(r"\n\s*\n", text):
            block = block.strip()
            if not block:
                continue
            first_line, _, rest = block.partition("\n")
            m = HEADING_RE.match(first_line) if doc.is_markdown else None
            if m:
                if paragraphs:
                    sections.append((heading, paragraphs))
                    paragraphs = []
                heading = m.group(2).strip()
                if rest.strip():
                    paragraphs.append(rest.strip())
                continue
            paragraphs.append(re.sub(r"[ \t]+", " ", block))
        if paragraphs:
            sections.append((heading, paragraphs))

        if doc.is_markdown and len(sections) >= 3 and page_no is None:
            chunks.append(_overview_chunk(title, sections, len(chunks)))

        for heading, paras in sections:
            for group in _pack(paras, max_chars, overlap):
                crumb = f"{title} > {heading}" if heading else title
                if page_no:
                    crumb += f" (page {page_no})"
                chunks.append(
                    Chunk(
                        text=f"{crumb}\n" + "\n\n".join(group),
                        section=_section_label(group, heading),
                        heading=heading,
                        index=len(chunks),
                        page=page_no,
                    )
                )
    return chunks
