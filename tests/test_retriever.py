from rag.chunking import Chunk
from rag.retriever import HybridRetriever, tokenize

TEXTS = {
    "kruskal": "Kruskal's algorithm sorts all edges by weight and adds the lightest edge that does not form a cycle, using union find.",
    "prim": "Prim's algorithm grows one tree from a start vertex, repeatedly adding the cheapest edge leaving the tree, using a priority queue.",
    "boruvka": "Borůvka's algorithm lets every component pick its cheapest outgoing edge in parallel rounds.",
    "cut": "The cut property: the minimum weight edge crossing any cut belongs to some minimum spanning tree.",
}


def index(store, embedder):
    chunks = [
        Chunk(name, text, {"source": "notes.pdf", "page": n, "page_end": n, "section": name, "algorithms": "", "chunk_index": n})
        for n, (name, text) in enumerate(TEXTS.items(), start=1)
    ]
    store.replace_source("notes.pdf", chunks, embedder.embed_documents([c.text for c in chunks]))


def test_tokenize_strips_accents():
    assert tokenize("Borůvka's O(E log V)") == ["boruvka", "s", "o", "e", "log", "v"]


def test_hybrid_retrieval_ranks_matching_chunk_first(store, embedder):
    index(store, embedder)
    retriever = HybridRetriever(store, embedder, candidate_k=4)

    result = retriever.retrieve("How does Boruvka pick the cheapest outgoing edge?", top_k=2)
    assert result.chunks[0].chunk.id == "boruvka"
    assert len(result.chunks) == 2
    assert result.chunks[0].chunk.metadata["page"] == 3
    assert 0 < result.best_similarity <= 1


def test_metadata_survives_round_trip_without_page(store, embedder):
    chunk = Chunk("md", "Markdown note about Prim.", {"source": "extra.md", "page": None, "page_end": None,
                                                       "section": "", "algorithms": "Prim", "chunk_index": 0})
    store.replace_source("extra.md", [chunk], embedder.embed_documents([chunk.text]))
    [(stored, _)] = store.query(embedder.embed_query("Prim"), k=1)
    assert stored.metadata["page"] is None
    assert stored.metadata["algorithms"] == "Prim"


def test_replace_source_does_not_duplicate(store, embedder):
    index(store, embedder)
    index(store, embedder)
    assert store.count() == len(TEXTS)


def test_reranker_reorders(store, embedder):
    index(store, embedder)

    class PreferCut:
        def predict(self, pairs):
            return [10.0 if "cut property" in text else 0.0 for _, text in pairs]

    retriever = HybridRetriever(store, embedder, candidate_k=4, reranker=PreferCut())
    result = retriever.retrieve("Kruskal sorts edges", top_k=1)
    assert result.chunks[0].chunk.id == "cut"


def test_empty_index(store, embedder):
    retriever = HybridRetriever(store, embedder)
    result = retriever.retrieve("anything")
    assert result.chunks == [] and result.best_similarity == 0.0
