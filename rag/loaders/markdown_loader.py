import re
from pathlib import Path

from rag.loaders.base import BaseLoader, Document
from rag.preprocessing import clean_text

_SOURCE = re.compile(r"<!--\s*source:\s*(.+?)\s*-->")
_PAGE = re.compile(r"<!--\s*page\s+(\d+)\s*-->")
_OTHER_COMMENT = re.compile(r"<!--(?!\s*page\s+\d+\s*-->).*?-->", re.S)


class MarkdownLoader(BaseLoader):
    """Markdown and plain text. Markdown headings are kept as they are.

    Reviewed notes from scripts/prepare_notes.py carry `<!-- source: notes.pdf -->` (used as the
    citation source) and `<!-- page N -->` markers (split into one Document per page).
    """

    extensions = (".md", ".markdown", ".txt")

    def load(self, path: Path) -> list[Document]:
        raw = path.read_text(encoding="utf-8")
        match = _SOURCE.search(raw)
        source = match.group(1) if match else path.name
        parts = _PAGE.split(_OTHER_COMMENT.sub("", raw))

        # parts = [text before the first marker, page, text, page, text, ...]
        pages = [(None, parts[0])] + [(int(n), text) for n, text in zip(parts[1::2], parts[2::2])]
        docs = []
        for page, text in pages:
            text = clean_text(text)
            if text:
                docs.append(Document(text=text, metadata={"source": source, "page": page}))
        return docs
