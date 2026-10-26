from pathlib import Path

from rag.loaders.base import BaseLoader, Document
from rag.loaders.docx_loader import DocxLoader
from rag.loaders.markdown_loader import MarkdownLoader
from rag.loaders.pdf_loader import PDFLoader

# To support a new format, write a BaseLoader subclass and add it here.
_LOADERS: list[BaseLoader] = [PDFLoader(), MarkdownLoader(), DocxLoader()]
_BY_EXTENSION = {ext: loader for loader in _LOADERS for ext in loader.extensions}


def supported_extensions() -> set[str]:
    return set(_BY_EXTENSION)


def load_file(path: Path) -> list[Document]:
    loader = _BY_EXTENSION.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"No loader for {path.suffix!r} files ({path.name})")
    return loader.load(path)


def load_directory(directory: Path) -> list[Document]:
    docs = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in _BY_EXTENSION:
            docs.extend(load_file(path))
    return docs


__all__ = ["BaseLoader", "Document", "load_directory", "load_file", "supported_extensions"]
