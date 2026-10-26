import logging
import time
from collections.abc import Iterator

from rag.llm.base import LLMClient, LLMRateLimitError

log = logging.getLogger(__name__)


class FallbackLLM(LLMClient):
    """Tries providers in order (e.g. Groq, then Gemini). Each free tier has its own quota, so
    when one is rate-limited the next answers. A rate-limited provider is skipped for
    `cooldown_seconds` so later questions don't wait on a refusal we already expect.

    A provider is only abandoned before it has produced any text; once an answer is streaming,
    an error ends the answer rather than restarting it with a different model.
    """

    def __init__(self, clients: list[LLMClient], cooldown_seconds: float = 60, clock=time.monotonic):
        first = clients[0]
        super().__init__(model=" -> ".join(c.model for c in clients), temperature=first.temperature, max_tokens=first.max_tokens)
        self.clients = clients
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self._cooling_until: dict[int, float] = {}

    def stream(self, system: str, messages: list[dict], max_tokens: int | None = None) -> Iterator[str]:
        now = self.clock()
        ready = [i for i in range(len(self.clients)) if self._cooling_until.get(i, 0) <= now]
        order = ready or list(range(len(self.clients)))  # everyone cooling down: try them all anyway
        rate_limited, last_error = False, None
        for i in order:
            client, started = self.clients[i], False
            try:
                for token in client.stream(system, messages, max_tokens):
                    started = True
                    yield token
                return
            except LLMRateLimitError as exc:
                if started:
                    raise
                log.warning("%s is rate-limited; cooling down for %ss", client.model, self.cooldown_seconds)
                self._cooling_until[i] = self.clock() + self.cooldown_seconds
                rate_limited, last_error = True, exc
            except Exception as exc:
                if started:
                    raise
                log.exception("%s failed; trying the next provider", client.model)
                last_error = exc
        if rate_limited:
            raise LLMRateLimitError("Every LLM provider is rate-limited") from last_error
        raise last_error
