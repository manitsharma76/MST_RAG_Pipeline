"""Hybrid retrieval: BM25 keyword search + dense vector search, fused with Reciprocal Rank Fusion."""
import re
import unicodedata
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from rag.chunking import Chunk
from rag.embeddings import Embedder
from rag.search_text import to_search_text
from rag.vectorstore import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # RRF score, or cross-encoder score when reranking is on
    similarity: float | None  # dense cosine similarity, if the chunk was a dense hit


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    best_similarity: float  # highest dense similarity; used by the grounding gate


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents (Borůvka -> boruvka), keep alphanumeric tokens."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.findall(r"[a-z0-9]+", text)


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, candidate_k: int = 20, rrf_k: int = 60, reranker=None):
        self.store = store
        self.embedder = embedder
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the BM25 index from the chunks stored in Chroma (call after re-indexing)."""
        self.chunks = self.store.all_chunks()
        self._bm25 = BM25Okapi([tokenize(to_search_text(c.text)) for c in self.chunks]) if self.chunks else None

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        search_query = to_search_text(query)
        dense = self.store.query(self.embedder.embed_query(search_query), self.candidate_k)
        sparse = self._bm25_search(search_query)

        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        similarity: dict[str, float] = {}
        for rank, (chunk, sim) in enumerate(dense):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            by_id[chunk.id] = chunk
            similarity[chunk.id] = sim
        for rank, chunk in enumerate(sparse):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            by_id[chunk.id] = chunk

        ranked = sorted(fused, key=fused.get, reverse=True)
        if self.reranker is not None and ranked:
            ranked = ranked[: self.candidate_k]
            scores = self.reranker.predict([(query, by_id[i].text) for i in ranked])
            fused = dict(zip(ranked, (float(s) for s in scores)))
            ranked = sorted(ranked, key=fused.get, reverse=True)

        results = [RetrievedChunk(by_id[i], fused[i], similarity.get(i)) for i in ranked[:top_k]]
        best = max((sim for _, sim in dense), default=0.0)
        return RetrievalResult(results, best)

    def _bm25_search(self, query: str) -> list[Chunk]:
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in order[: self.candidate_k] if scores[i] > 0]
