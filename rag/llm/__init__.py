import logging
import os

from rag.llm.base import LLMClient, LLMRateLimitError

log = logging.getLogger(__name__)
KEY_NAMES = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}


class MissingAPIKeyError(RuntimeError):
    pass


def get_llm(settings) -> LLMClient:
    """The main provider (LLM_PROVIDER), followed by any LLM_BACKUP_PROVIDERS that have an API key.
    With more than one, a FallbackLLM moves to the next provider when one is rate-limited."""
    main_name = settings.llm_provider.strip().lower()
    main = _client(main_name, settings)
    if main is None:
        raise MissingAPIKeyError(f"{KEY_NAMES[main_name]} is not set. Add it to .env locally, or to the host's environment.")

    backup_names = [n.strip().lower() for n in settings.llm_backup_providers.split(",") if n.strip()]
    backups = [c for c in (_client(n, settings) for n in backup_names if n != main_name) if c is not None]
    if not backups:
        return main
    from rag.llm.fallback import FallbackLLM

    chain = FallbackLLM([main, *backups])
    log.info("LLM chain: %s", chain.model)
    return chain


def _client(provider: str, settings) -> LLMClient | None:
    """A client for `provider`, or None when its API key isn't set."""
    if provider not in KEY_NAMES:
        raise ValueError(f"Unknown LLM provider {provider!r}; use 'groq' or 'gemini'")
    key = os.getenv(KEY_NAMES[provider])
    if not key:
        return None
    common = {"temperature": settings.temperature, "max_tokens": settings.max_tokens}
    if provider == "groq":
        from rag.llm.groq_client import GroqClient

        fallbacks = [m.strip() for m in settings.groq_fallback_models.split(",") if m.strip()]
        return GroqClient(api_key=key, model=settings.groq_model, fallback_models=fallbacks, **common)
    from rag.llm.gemini_client import GeminiClient

    return GeminiClient(api_key=key, model=settings.gemini_model, **common)


__all__ = ["LLMClient", "LLMRateLimitError", "MissingAPIKeyError", "get_llm"]
