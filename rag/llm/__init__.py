import os

from rag.llm.base import LLMClient


class MissingAPIKeyError(RuntimeError):
    pass


def get_llm(settings) -> LLMClient:
    provider = settings.llm_provider.lower()
    common = {"temperature": settings.temperature, "max_tokens": settings.max_tokens}

    if provider == "groq":
        from rag.llm.groq_client import GroqClient

        return GroqClient(api_key=_require("GROQ_API_KEY"), model=settings.groq_model, **common)
    if provider == "gemini":
        from rag.llm.gemini_client import GeminiClient

        return GeminiClient(api_key=_require("GEMINI_API_KEY"), model=settings.gemini_model, **common)
    raise ValueError(f"Unknown LLM_PROVIDER {settings.llm_provider!r}; use 'groq' or 'gemini'")


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise MissingAPIKeyError(f"{name} is not set. Add it to .env locally, or to the Space's Settings → Secrets.")
    return value


__all__ = ["LLMClient", "MissingAPIKeyError", "get_llm"]
