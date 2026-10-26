from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """One unit of loaded text (a PDF page, or a whole Markdown/DOCX file).

    Text is normalised to Markdown-style headings (`# Title`, `## Section`) so the
    chunker can treat every format the same way.
    metadata keys: source (file name), page (1-based, or None).
    """

    text: str
    metadata: dict = field(default_factory=dict)


class BaseLoader(ABC):
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        ...
