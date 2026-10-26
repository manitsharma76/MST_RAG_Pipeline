from collections import Counter
from pathlib import Path
from statistics import median

import pymupdf

from rag.loaders.base import BaseLoader, Document
from rag.preprocessing import clean_text, find_repeated_lines, normalize_line


class PDFLoader(BaseLoader):
    """Extracts text page by page and marks headings using font size and boldness."""

    extensions = (".pdf",)

    def load(self, path: Path) -> list[Document]:
        with pymupdf.open(path) as pdf:
            pages = [page.get_text("dict") for page in pdf]

        body_size = self._body_font_size(pages)
        page_lines = [self._page_lines(page, body_size) for page in pages]
        boilerplate = find_repeated_lines(page_lines)

        docs = []
        for number, lines in enumerate(page_lines, start=1):
            kept = [line for line in lines if not line.strip() or normalize_line(line) not in boilerplate]
            text = clean_text("\n".join(kept))
            if text:
                docs.append(Document(text=text, metadata={"source": path.name, "page": number}))
        return docs

    @staticmethod
    def _spans(page: dict):
        for block in page["blocks"]:
            for line in block.get("lines", []):
                yield line

    def _body_font_size(self, pages: list[dict]) -> float:
        sizes = Counter()
        for page in pages:
            for line in self._spans(page):
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes[round(span["size"], 1)] += len(span["text"])
        return sizes.most_common(1)[0][0] if sizes else 11.0

    def _page_lines(self, page: dict, body_size: float) -> list[str]:
        lines = []
        for block in page["blocks"]:
            if "lines" not in block:  # image block
                continue
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in line["spans"]).rstrip()
                size = median(s["size"] for s in spans)
                bold = all(s["flags"] & 16 for s in spans)  # bit 4 = bold
                prefix = self._heading_prefix(text, size, bold, body_size)
                previous = next((i for i in range(len(lines) - 1, -1, -1) if lines[i]), None)
                if prefix and previous is not None and lines[previous].startswith(prefix):
                    # Heading wrapped onto a second line (possibly a separate text block).
                    del lines[previous + 1:]
                    lines[previous] += " " + text.strip()
                else:
                    lines.append(prefix + text)
            lines.append("")  # blank line between blocks keeps paragraphs apart
        return lines

    @staticmethod
    def _heading_prefix(text: str, size: float, bold: bool, body_size: float) -> str:
        short = len(text) <= 80 and not text.rstrip().endswith((".", ",", ";"))
        if not short:
            return ""
        if size >= body_size * 1.5:
            return "# "
        if size >= body_size * 1.15:
            return "## "
        if bold and len(text.split()) <= 10:
            return "### "
        return ""
