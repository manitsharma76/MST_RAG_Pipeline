import logging
from collections.abc import Iterator

from rag.llm.base import TRUNCATED_NOTE, LLMClient, LLMRateLimitError

log = logging.getLogger(__name__)


class GroqClient(LLMClient):
    """Groq chat completions. When a model is rate-limited (free-tier per-minute token limits
    move with load), the request is retried on each fallback model, which has its own quota."""

    def __init__(self, api_key: str, fallback_models: list[str] | tuple = (), **kwargs):
        super().__init__(**kwargs)
        from groq import Groq

        self.client = Groq(api_key=api_key, max_retries=1)
        self.fallback_models = list(fallback_models)

    def stream(self, system: str, messages: list[dict], max_tokens: int | None = None) -> Iterator[str]:
        from groq import RateLimitError

        models = [self.model, *self.fallback_models]
        for attempt, model in enumerate(models):
            extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
            try:
                # Rate limits are reported when the request is made, before any token streams.
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, *messages],
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    stream=True,
                    **extra,
                )
            except RateLimitError as exc:
                log.warning("Groq rate limit on %s: %s", model, exc)
                if attempt == len(models) - 1:
                    raise LLMRateLimitError(f"All Groq models are rate-limited ({', '.join(models)})") from exc
                continue
            finish_reason = None
            for event in response:
                if not event.choices:
                    continue
                if event.choices[0].delta.content:
                    yield event.choices[0].delta.content
                finish_reason = getattr(event.choices[0], "finish_reason", None) or finish_reason
            if finish_reason == "length":  # hit max_tokens mid-answer
                yield TRUNCATED_NOTE
            return
