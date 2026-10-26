from collections.abc import Iterator

from rag.llm.base import LLMClient


class GroqClient(LLMClient):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        from groq import Groq

        self.client = Groq(api_key=api_key)

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for event in response:
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                yield delta
