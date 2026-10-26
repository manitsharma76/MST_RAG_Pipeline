from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Runs the embedding model on the local CPU with ONNX Runtime (no PyTorch).

    Same BGE weights as sentence-transformers and identical vectors, at under half the memory,
    so the app fits small free hosting instances. BGE models expect an instruction prefix on
    queries but not on documents.
    """

    def __init__(self, model_name: str, query_instruction: str = "", cache_dir: Path | None = None):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name, cache_dir=str(self.cache_dir) if self.cache_dir else None)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.embed([self.query_instruction + text]))).tolist()

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_pretrained(self.model_name)
        return len(self._tokenizer.encode(text).ids)


class FastEmbedReranker:
    """Cross-encoder reranker on ONNX Runtime, with the `predict(pairs)` interface the retriever uses."""

    def __init__(self, model_name: str, cache_dir: Path | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name, cache_dir=str(cache_dir) if cache_dir else None)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        return [float(score) for score in self.model.rerank(pairs[0][0], [text for _, text in pairs])]
