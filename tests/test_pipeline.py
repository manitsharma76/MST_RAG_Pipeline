from rag.chunking import Chunk
from rag.llm.base import LLMClient
from rag.pipeline import RAGPipeline, cited_indices
from rag.prompts import NOT_COVERED, SYSTEM_PROMPT
from rag.retriever import RetrievalResult, RetrievedChunk

CHUNK = RetrievedChunk(
    Chunk("c1", "Kruskal's algorithm runs in O(E log E).", {"source": "notes.pdf", "page": 4, "page_end": 4,
                                                            "section": "Kruskal", "algorithms": "Kruskal", "chunk_index": 0}),
    score=0.03,
    similarity=0.8,
)


class FakeRetriever:
    def __init__(self, similarity=0.8):
        self.similarity = similarity
        self.queries = []

    def retrieve(self, query, top_k=5):
        self.queries.append(query)
        return RetrievalResult([CHUNK], self.similarity)


class FakeLLM(LLMClient):
    def __init__(self, reply="Kruskal runs in O(E log E) [1].", rewrite="What is the complexity of Kruskal's algorithm?", fail=False):
        super().__init__(model="fake")
        self.reply, self.rewrite, self.fail = reply, rewrite, fail
        self.calls = []

    def stream(self, system, messages):
        self.calls.append((system, messages))
        if self.fail:
            raise RuntimeError("rate limited")
        text = self.reply if system == SYSTEM_PROMPT else self.rewrite
        for word in text.split(" "):
            yield word + " "


def test_answer_streams_with_context_and_citations():
    llm = FakeLLM()
    pipeline = RAGPipeline(FakeRetriever(), llm, min_similarity=0.5)
    steps = list(pipeline.stream_answer("Complexity of Kruskal?"))
    assert len(steps) > 1
    answer, sources = steps[-1]
    assert answer.strip() == "Kruskal runs in O(E log E) [1]."
    assert sources == [CHUNK]
    system, messages = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "[1] (notes.pdf, p. 4, Kruskal)" in messages[0]["content"]


def test_grounding_gate_refuses_without_calling_llm():
    llm = FakeLLM()
    pipeline = RAGPipeline(FakeRetriever(similarity=0.2), llm, min_similarity=0.5)
    answer, _ = pipeline.answer("Explain Dijkstra's algorithm")
    assert answer == NOT_COVERED
    assert llm.calls == []


def test_follow_up_is_rewritten_before_retrieval():
    retriever, llm = FakeRetriever(), FakeLLM()
    pipeline = RAGPipeline(retriever, llm, min_similarity=0.5)
    history = [{"role": "user", "content": "Explain Kruskal's algorithm"}, {"role": "assistant", "content": "..."}]
    pipeline.answer("What about its complexity?", history)
    assert retriever.queries == ["What is the complexity of Kruskal's algorithm?"]


def test_no_rewrite_on_first_turn():
    retriever, llm = FakeRetriever(), FakeLLM()
    RAGPipeline(retriever, llm, min_similarity=0.5).answer("What is an MST?")
    assert retriever.queries == ["What is an MST?"]
    assert len(llm.calls) == 1


def test_llm_failure_is_reported_not_raised():
    pipeline = RAGPipeline(FakeRetriever(), FakeLLM(fail=True), min_similarity=0.5)
    answer, sources = pipeline.answer("Complexity of Kruskal?")
    assert "rate limited" in answer and sources == [CHUNK]


def test_missing_llm_returns_passages():
    answer, sources = RAGPipeline(FakeRetriever(), None, min_similarity=0.5).answer("Kruskal?")
    assert "most relevant passages" in answer and sources == [CHUNK]


def test_cited_indices():
    assert cited_indices("A [1]. B [2][3]. C [1].") == {1, 2, 3}
