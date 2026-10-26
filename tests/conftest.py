import hashlib
import math
import uuid

import chromadb
import pytest

from rag.retriever import tokenize
from rag.vectorstore import VectorStore


class FakeEmbedder:
    """Hashed bag-of-words vectors: deterministic, offline, and good enough for overlap-based tests."""

    dim = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def store():
    return VectorStore(f"test_{uuid.uuid4().hex[:8]}", client=chromadb.EphemeralClient())
