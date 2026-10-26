from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMClient(ABC):
    """Chat model behind a provider-neutral interface.

    `messages` are {"role": "user" | "assistant", "content": str} dicts; the system
    prompt is passed separately because providers handle it differently.
    """

    def __init__(self, model: str, temperature: float = 0.1, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        ...

    def generate(self, system: str, messages: list[dict]) -> str:
        return "".join(self.stream(system, messages))
