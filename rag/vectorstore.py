from pathlib import Path

import chromadb

from rag.chunking import Chunk


class VectorStore:
    """Embedded ChromaDB collection. Runs in-process and persists to a folder; no server."""

    def __init__(self, collection_name: str, path: Path | None = None, client=None):
        if client is None:
            settings = chromadb.Settings(anonymized_telemetry=False)
            client = chromadb.PersistentClient(path=str(path), settings=settings)
        self.client = client
        self.collection_name = collection_name
        self.collection = self._get_collection()

    def _get_collection(self):
        return self.client.get_or_create_collection(self.collection_name, metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_collection()

    def replace_source(self, source: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Remove every chunk from `source`, then add the new ones (used for re-ingesting a file)."""
        self.collection.delete(where={"source": source})
        if chunks:
            self.collection.add(
                ids=[c.id for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[_to_chroma(c.metadata) for c in chunks],
                embeddings=embeddings,
            )

    def sources(self) -> set[str]:
        return {m["source"] for m in self.collection.get(include=["metadatas"])["metadatas"]}

    def all_chunks(self) -> list[Chunk]:
        data = self.collection.get(include=["documents", "metadatas"])
        chunks = [Chunk(i, d, _from_chroma(m)) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])]
        return sorted(chunks, key=lambda c: (c.metadata["source"], c.metadata["chunk_index"]))

    def query(self, embedding: list[float], k: int, where: dict | None = None) -> list[tuple[Chunk, float]]:
        """Return (chunk, cosine similarity) pairs, most similar first."""
        k = min(k, self.count())
        if k == 0:
            return []
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            (Chunk(i, d, _from_chroma(m)), 1.0 - dist)
            for i, d, m, dist in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]


def _to_chroma(metadata: dict) -> dict:
    # Chroma metadata values cannot be None.
    return {k: v for k, v in metadata.items() if v is not None}


def _from_chroma(metadata: dict) -> dict:
    return {"page": None, "page_end": None, "section": "", "algorithms": "", **metadata}
