from dataclasses import dataclass
from pathlib import Path

from rag.chunking import Chunk, chunk_documents
from rag.embeddings import Embedder
from rag.loaders import Document, load_file, supported_extensions
from rag.search_text import to_search_text
from rag.vectorstore import VectorStore


@dataclass
class IndexStats:
    files: int
    chunks: list[Chunk]


def note_files(raw_dir: Path, parsed_dir: Path | None = None) -> list[Path]:
    """Files to index. A reviewed `parsed_dir/<stem>.md` (from scripts/prepare_notes.py) is used
    instead of the PDF with the same stem; PDFs without one fall back to the PDF loader."""
    reviewed = {p.stem: p for p in sorted(parsed_dir.glob("*.md"))} if parsed_dir and parsed_dir.exists() else {}
    files = []
    for path in sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_extensions()):
        files.append(reviewed.pop(path.stem) if path.suffix.lower() == ".pdf" and path.stem in reviewed else path)
    return files + list(reviewed.values())


def load_notes(raw_dir: Path, parsed_dir: Path | None = None) -> list[tuple[str, list[Document]]]:
    """(source name, documents) for every note file. The source name is what citations show."""
    notes = []
    for path in note_files(raw_dir, parsed_dir):
        docs = load_file(path)
        notes.append((docs[0].metadata["source"] if docs else path.name, docs))
    return notes


def embed_chunks(embedder: Embedder, chunks: list[Chunk]) -> list[list[float]]:
    """Embed the search view of each chunk (symbols spelled out); the stored text stays exact."""
    return embedder.embed_documents([to_search_text(c.text) for c in chunks])


def build_index(raw_dir: Path, store: VectorStore, embedder: Embedder, chunk_size: int, chunk_overlap: int,
                rebuild: bool = False, parsed_dir: Path | None = None) -> IndexStats:
    """Ingest every note. Each source's old chunks are replaced, so re-running after editing or
    adding notes is safe. `rebuild` also drops sources that were removed."""
    if rebuild:
        store.reset()

    notes = load_notes(raw_dir, parsed_dir)
    all_chunks = []
    for source, docs in notes:
        chunks = chunk_documents(docs, chunk_size, chunk_overlap)
        store.replace_source(source, chunks, embed_chunks(embedder, chunks))
        all_chunks.extend(chunks)

    for stale in store.sources() - {source for source, _ in notes}:
        store.replace_source(stale, [], [])

    return IndexStats(files=len(notes), chunks=all_chunks)
