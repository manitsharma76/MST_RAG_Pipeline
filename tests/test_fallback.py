import pytest

from rag.llm.base import LLMClient, LLMRateLimitError
from rag.llm.fallback import FallbackLLM


class Provider(LLMClient):
    def __init__(self, name, outcome):
        super().__init__(model=name)
        self.outcome, self.calls = outcome, 0

    def stream(self, system, messages, max_tokens=None):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        yield from self.outcome


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_uses_the_main_provider_when_it_works():
    groq, gemini = Provider("groq", ["from ", "groq"]), Provider("gemini", ["from gemini"])
    assert FallbackLLM([groq, gemini]).generate("s", []) == "from groq"
    assert gemini.calls == 0


def test_rate_limited_provider_falls_back_and_cools_down():
    clock = Clock()
    groq, gemini = Provider("groq", LLMRateLimitError("quota")), Provider("gemini", ["from gemini"])
    llm = FallbackLLM([groq, gemini], cooldown_seconds=60, clock=clock)
    assert llm.generate("s", []) == "from gemini"
    assert llm.generate("s", []) == "from gemini"
    assert groq.calls == 1  # skipped while cooling down
    clock.now = 61
    llm.generate("s", [])
    assert groq.calls == 2  # tried again after the cool-down


def test_all_rate_limited_raises_rate_limit():
    llm = FallbackLLM([Provider("groq", LLMRateLimitError("a")), Provider("gemini", LLMRateLimitError("b"))])
    with pytest.raises(LLMRateLimitError):
        llm.generate("s", [])


def test_other_errors_also_fall_back_but_surface_if_nothing_works():
    gemini = Provider("gemini", ["ok"])
    assert FallbackLLM([Provider("groq", RuntimeError("network")), gemini]).generate("s", []) == "ok"
    with pytest.raises(RuntimeError, match="down"):
        FallbackLLM([Provider("groq", RuntimeError("network")), Provider("gemini", RuntimeError("down"))]).generate("s", [])


def test_error_after_text_started_is_not_retried_elsewhere():
    class Breaks(Provider):
        def stream(self, system, messages, max_tokens=None):
            yield "partial "
            raise LLMRateLimitError("mid-stream")

    gemini = Provider("gemini", ["full"])
    with pytest.raises(LLMRateLimitError):
        "".join(FallbackLLM([Breaks("groq", None), gemini]).stream("s", []))
    assert gemini.calls == 0


def test_get_llm_adds_backup_only_when_its_key_is_set(monkeypatch):
    from config import settings
    from rag.llm import get_llm
    from rag.llm.groq_client import GroqClient

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(get_llm(settings), GroqClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    chain = get_llm(settings)
    assert isinstance(chain, FallbackLLM) and [type(c).__name__ for c in chain.clients] == ["GroqClient", "GeminiClient"]
