"""Heading-aware chunking.

Documents are split into sections at Markdown headings, sections into paragraphs
(blocks separated by a blank line), and paragraphs are packed into chunks of at most
`chunk_size` characters. Paragraphs are never split unless a single one is larger than
a chunk, so pseudocode and tables that form one block stay together. A fenced code block
is always one paragraph, even with blank lines inside, and if it must be split each piece
is re-fenced; an oversized table repeats its header row in each piece.
"""
import hashlib
import re
from dataclasses import dataclass, field
from itertools import groupby

from rag.loaders.base import Document

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

ALGORITHMS = {
    "kruskal": "Kruskal",
    "prim": "Prim",
    "boruvka": "Boruvka",
    "borůvka": "Boruvka",
    "reverse-delete": "Reverse-Delete",
    "reverse delete": "Reverse-Delete",
    "union-find": "Union-Find",
    "union find": "Union-Find",
    "disjoint set": "Union-Find",
}


# Headings that only make sense under a topic ("Runtime" of what?). One is nested under the
# nearest enclosing topic heading such as "Prim's Algorithm", whatever level the notes gave it.
GENERIC_HEADINGS = {
    "analysis", "complexity", "correctness", "example", "examples", "implementation",
    "proof", "proof of correctness", "pseudocode", "running time", "runtime", "time complexity",
}


def _is_generic(title: str) -> bool:
    lowered = title.lower().strip()
    return lowered in GENERIC_HEADINGS or lowered.endswith(("pseudocode", "psuedocode"))


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class _Paragraph:
    text: str
    page: int | None


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks = []
    for source, source_docs in groupby(docs, key=lambda d: d.metadata["source"]):
        index = 0
        for section, paragraphs in _sections(list(source_docs)):
            for body in _pack(paragraphs, chunk_size, chunk_overlap):
                pages = [p.page for p in body if p.page is not None]
                text = "\n\n".join(p.text for p in body)
                if section:
                    text = f"{section}\n\n{text}"
                chunks.append(
                    Chunk(
                        id=hashlib.sha1(f"{source}:{index}".encode()).hexdigest()[:16],
                        text=text,
                        metadata={
                            "source": source,
                            "page": min(pages) if pages else None,
                            "page_end": max(pages) if pages else None,
                            "section": section,
                            "algorithms": _detect_algorithms(text),
                            "chunk_index": index,
                        },
                    )
                )
                index += 1
    return chunks


def _sections(docs: list[Document]):
    """Yield (heading path, paragraphs). Sections can span several pages."""
    path: list[tuple[int, str]] = []  # (heading level, title); levels may skip, e.g. ## with no #
    paragraphs: list[_Paragraph] = []
    current: list[str] = []

    def close_paragraph(page):
        if current:
            paragraphs.append(_Paragraph("\n".join(current).strip(), page))
            current.clear()

    page = None
    in_fence = False
    for doc in docs:
        if not in_fence:  # a code block continuing onto the next page stays one paragraph
            close_paragraph(page)
        page = doc.metadata.get("page")
        for line in doc.text.splitlines():
            if line.strip().startswith("```"):
                in_fence = not in_fence
                current.append(line)
                continue
            if in_fence:  # blank lines and "#" comments inside code are code
                current.append(line)
                continue
            heading = _HEADING.match(line)
            if heading:
                close_paragraph(page)
                if paragraphs:
                    yield _path_label(path), paragraphs
                    paragraphs = []
                path = _enter_heading(path, len(heading.group(1)), heading.group(2).strip())
            elif line.strip():
                current.append(line)
            else:
                close_paragraph(page)
    close_paragraph(page)
    if paragraphs:
        yield _path_label(path), paragraphs


def _enter_heading(path: list[tuple[int, str]], level: int, title: str) -> list[tuple[int, str]]:
    if _is_generic(title):
        # Nest under the nearest topic heading, replacing a generic sibling such as "Pseudocode".
        parent = list(path)
        while parent and _is_generic(parent[-1][1]):
            parent.pop()
        if parent:
            return parent + [(parent[-1][0] + 1, title)]
    return [entry for entry in path if entry[0] < level] + [(level, title)]


def _path_label(path: list[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in path)


def _pack(paragraphs: list[_Paragraph], chunk_size: int, chunk_overlap: int):
    """Greedily pack paragraphs into chunks, carrying trailing paragraphs as overlap."""
    pieces = [piece for p in paragraphs for piece in _split_oversized(p, chunk_size)]
    chunk: list[_Paragraph] = []
    for piece in pieces:
        if chunk and _length(chunk + [piece]) > chunk_size:
            yield chunk
            chunk = _tail(chunk, chunk_overlap)
            while chunk and _length(chunk + [piece]) > chunk_size:
                chunk = chunk[1:]
        chunk.append(piece)
    if chunk:
        yield chunk


def _split_oversized(paragraph: _Paragraph, chunk_size: int) -> list[_Paragraph]:
    if len(paragraph.text) <= chunk_size:
        return [paragraph]
    lines = paragraph.text.splitlines()
    if lines[0].strip().startswith("```"):
        body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        return _split_lines(body, chunk_size, paragraph.page, prefix=[lines[0]], suffix=["```"])
    if len(lines) > 2 and lines[0].startswith("|") and re.match(r"^\|[\s:|-]+\|$", lines[1]):
        return _split_lines(lines[2:], chunk_size, paragraph.page, prefix=lines[:2])
    return _split_text(paragraph, chunk_size)


def _split_lines(lines: list[str], chunk_size: int, page, prefix: list[str], suffix: list[str] = ()) -> list[_Paragraph]:
    """Split by whole lines, wrapping every piece in `prefix`/`suffix` (code fence or table header)."""
    budget = max(chunk_size - sum(len(x) + 1 for x in [*prefix, *suffix]), 1)
    pieces, current = [], []
    for line in lines:
        if current and sum(len(x) + 1 for x in current) + len(line) > budget:
            pieces.append(current)
            current = []
        current.append(line)
    if current:
        pieces.append(current)
    return [_Paragraph("\n".join([*prefix, *piece, *suffix]), page) for piece in pieces]


def _split_text(paragraph: _Paragraph, chunk_size: int) -> list[_Paragraph]:
    # Prefer line breaks (keeps pseudocode lines whole), fall back to sentences.
    units = paragraph.text.splitlines()
    if max(len(u) for u in units) > chunk_size:
        units = re.split(r"(?<=[.!?])\s+", paragraph.text)
    pieces, current = [], ""
    for unit in units:
        candidate = f"{current}\n{unit}" if current else unit
        if current and len(candidate) > chunk_size:
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        pieces.append(current)
    return [
        _Paragraph(piece[start : start + chunk_size], paragraph.page)
        for piece in pieces
        for start in range(0, len(piece), chunk_size)
    ]


def _tail(chunk: list[_Paragraph], chunk_overlap: int) -> list[_Paragraph]:
    tail = []
    for paragraph in reversed(chunk):
        if _length(tail + [paragraph]) > chunk_overlap:
            break
        tail.insert(0, paragraph)
    return tail


def _length(paragraphs: list[_Paragraph]) -> int:
    return sum(len(p.text) for p in paragraphs) + 2 * max(len(paragraphs) - 1, 0)


def _detect_algorithms(text: str) -> str:
    lowered = text.lower()
    found = {name for key, name in ALGORITHMS.items() if re.search(rf"\b{re.escape(key)}\b", lowered)}
    return ", ".join(sorted(found))
