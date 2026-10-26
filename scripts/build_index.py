"""Ingest the notes in data/raw/ into the Chroma index.

    python scripts/build_index.py            # add or update files
    python scripts/build_index.py --rebuild  # start from an empty collection
    python scripts/build_index.py --show 5   # also print the first 5 chunks
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from rag.embeddings import FastEmbedEmbedder  # noqa: E402
from rag.indexing import build_index, note_files  # noqa: E402
from rag.prompts import citation_label  # noqa: E402
from rag.retriever import RetrievedChunk  # noqa: E402
from rag.vectorstore import VectorStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="delete the collection before indexing")
    parser.add_argument("--show", type=int, default=3, help="number of sample chunks to print")
    args = parser.parse_args()

    if not settings.raw_dir.exists() or not any(settings.raw_dir.iterdir()):
        sys.exit(f"No notes found. Put your PDF (or .md/.txt/.docx) files in {settings.raw_dir}")

    store = VectorStore(settings.collection_name, path=settings.index_dir / "chroma")
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.query_instruction, settings.model_cache_dir)
    stats = build_index(settings.raw_dir, store, embedder, settings.chunk_size, settings.chunk_overlap, args.rebuild,
                        parsed_dir=settings.parsed_dir)

    for path in note_files(settings.raw_dir, settings.parsed_dir):
        kind = "reviewed Markdown" if path.parent == settings.parsed_dir else "loaded directly"
        print(f"  {path.relative_to(settings.raw_dir.parents[1])}  ({kind})")

    sizes = [len(c.text) for c in stats.chunks]
    print(f"Indexed {stats.files} file(s) into {len(stats.chunks)} chunks "
          f"(chars: min {min(sizes, default=0)}, avg {sum(sizes) // max(len(sizes), 1)}, max {max(sizes, default=0)}).")
    print(f"Collection now holds {store.count()} chunks at {settings.index_dir / 'chroma'}")

    for chunk in stats.chunks[: args.show]:
        print("\n" + "=" * 80)
        print(f"[{citation_label(RetrievedChunk(chunk, 0.0, None))}]  algorithms: {chunk.metadata['algorithms'] or '-'}")
        print("-" * 80)
        print(chunk.text[:800] + (" …" if len(chunk.text) > 800 else ""))


if __name__ == "__main__":
    main()
