from collections.abc import Iterator

from rag.llm.base import TRUNCATED_NOTE, LLMClient, LLMRateLimitError


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        from google import genai

        self.client = genai.Client(api_key=api_key)

    def stream(self, system: str, messages: list[dict], max_tokens: int | None = None) -> Iterator[str]:
        from google.genai import errors, types

        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=max_tokens or self.max_tokens,
        )
        try:
            finish_reason = None
            for event in self.client.models.generate_content_stream(model=self.model, contents=contents, config=config):
                if event.text:
                    yield event.text
                if event.candidates and event.candidates[0].finish_reason:
                    finish_reason = event.candidates[0].finish_reason
            if finish_reason == types.FinishReason.MAX_TOKENS:
                yield TRUNCATED_NOTE
        except errors.ClientError as exc:
            if exc.code == 429:
                raise LLMRateLimitError(f"Gemini model {self.model} is rate-limited") from exc
            raise
