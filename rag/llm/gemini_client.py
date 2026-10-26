from collections.abc import Iterator

from rag.llm.base import LLMClient


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        from google import genai

        self.client = genai.Client(api_key=api_key)

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        from google.genai import types

        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        for event in self.client.models.generate_content_stream(model=self.model, contents=contents, config=config):
            if event.text:
                yield event.text
