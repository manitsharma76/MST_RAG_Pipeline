from abc import ABC, abstractmethod
from collections.abc import Iterator


TRUNCATED_NOTE = "…\n\n_(The answer was cut short. Ask about one part at a time for more detail.)_"


class LLMRateLimitError(RuntimeError):
    """The provider refused the request for quota reasons (every configured model was rate-limited)."""


class LLMClient(ABC):
    """Chat model behind a provider-neutral interface.

    `messages` are {"role": "user" | "assistant", "content": str} dicts; the system
    prompt is passed separately because providers handle it differently. `max_tokens`
    overrides the client default for one call (e.g. a short follow-up rewrite).
    """

    def __init__(self, model: str, temperature: float = 0.1, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def stream(self, system: str, messages: list[dict], max_tokens: int | None = None) -> Iterator[str]:
        ...

    def generate(self, system: str, messages: list[dict], max_tokens: int | None = None) -> str:
        return "".join(self.stream(system, messages, max_tokens))
