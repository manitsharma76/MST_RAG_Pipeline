from rag.chunking import Chunk
from rag.llm.base import LLMClient, LLMRateLimitError
from rag.pipeline import RAGPipeline, cited_indices
from rag.prompts import LLM_FAILED, NOT_COVERED, RATE_LIMITED, SYSTEM_PROMPT
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
    def __init__(self, reply="Kruskal runs in O(E log E) [1].", rewrite="What is the complexity of Kruskal's algorithm?", fail=None):
        super().__init__(model="fake")
        self.reply, self.rewrite, self.fail = reply, rewrite, fail
        self.calls = []

    def stream(self, system, messages, max_tokens=None):
        self.calls.append((system, messages, max_tokens))
        if self.fail:
            raise self.fail
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
    system, messages, _ = llm.calls[0]
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
    assert llm.calls[0][2] == 150  # the rewrite reserves few output tokens


def test_no_rewrite_on_first_turn():
    retriever, llm = FakeRetriever(), FakeLLM()
    RAGPipeline(retriever, llm, min_similarity=0.5).answer("What is an MST?")
    assert retriever.queries == ["What is an MST?"]
    assert len(llm.calls) == 1


def test_llm_failure_is_reported_without_details():
    pipeline = RAGPipeline(FakeRetriever(), FakeLLM(fail=RuntimeError("401 org_secret123")), min_similarity=0.5)
    answer, sources = pipeline.answer("Complexity of Kruskal?")
    assert answer == LLM_FAILED and "org_secret123" not in answer and sources == [CHUNK]


def test_rate_limit_gets_a_try_again_message():
    pipeline = RAGPipeline(FakeRetriever(), FakeLLM(fail=LLMRateLimitError("quota")), min_similarity=0.5)
    answer, _ = pipeline.answer("Complexity of Kruskal?")
    assert answer == RATE_LIMITED


def _groq_client(responses):
    """GroqClient wired to a fake SDK: `responses` maps model -> list of tokens, or an exception."""
    from types import SimpleNamespace

    from rag.llm.groq_client import GroqClient

    calls = []

    def create(model, **kwargs):
        calls.append((model, kwargs))
        outcome = responses[model]
        if isinstance(outcome, Exception):
            raise outcome
        events = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t), finish_reason=None)]) for t in outcome]
        events[-1].choices[0].finish_reason = "length" if outcome[-1].endswith("-") else "stop"
        return events

    client = GroqClient.__new__(GroqClient)
    LLMClient.__init__(client, model="qwen", max_tokens=700)
    client.fallback_models = ["openai/gpt-oss-120b"]
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls


def _rate_limit_error():
    import httpx
    from groq import RateLimitError

    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
    return RateLimitError("Request too large", response=response, body=None)


def test_groq_falls_back_to_next_model_on_rate_limit():
    client, calls = _groq_client({"qwen": _rate_limit_error(), "openai/gpt-oss-120b": ["Fallback ", "answer"]})
    assert client.generate("system", [{"role": "user", "content": "q"}]) == "Fallback answer"
    assert [model for model, _ in calls] == ["qwen", "openai/gpt-oss-120b"]
    assert calls[1][1]["reasoning_effort"] == "low" and calls[1][1]["max_tokens"] == 700


def test_groq_marks_answers_cut_off_by_max_tokens():
    from rag.llm.base import TRUNCATED_NOTE

    client, _ = _groq_client({"qwen": ["A long answer that stops mid-"]})
    assert client.generate("system", [{"role": "user", "content": "q"}]).endswith(TRUNCATED_NOTE)


def test_groq_raises_rate_limit_when_every_model_is_limited():
    import pytest

    client, _ = _groq_client({"qwen": _rate_limit_error(), "openai/gpt-oss-120b": _rate_limit_error()})
    with pytest.raises(LLMRateLimitError):
        client.generate("system", [{"role": "user", "content": "q"}], max_tokens=150)


def test_missing_llm_returns_passages():
    answer, sources = RAGPipeline(FakeRetriever(), None, min_similarity=0.5).answer("Kruskal?")
    assert "most relevant passages" in answer and sources == [CHUNK]


def test_cited_indices():
    assert cited_indices("A [1]. B [2][3]. C [1].") == {1, 2, 3}
