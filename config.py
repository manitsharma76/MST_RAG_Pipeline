"""Central configuration. Every value can be overridden with an environment variable."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent


def _env(name: str, default):
    value = os.getenv(name)
    if value is None:
        return default
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return type(default)(value)


@dataclass(frozen=True)
class Settings:
    # Paths
    raw_dir: Path = ROOT / "data" / "raw"
    parsed_dir: Path = ROOT / "data" / "parsed"  # reviewed Markdown from scripts/prepare_notes.py
    index_dir: Path = ROOT / "index"
    model_cache_dir: Path = ROOT / ".cache" / "fastembed"  # downloaded ONNX models
    collection_name: str = _env("COLLECTION_NAME", "mst_notes")

    # Chunking, in characters (~4 chars per token -> ~400 tokens, ~60 overlap). Kept under the
    # embedding model's 512-token limit, beyond which text is silently left out of the vector.
    # Compare sizes on your notes with scripts/compare_chunking.py.
    chunk_size: int = _env("CHUNK_SIZE", 1600)
    chunk_overlap: int = _env("CHUNK_OVERLAP", 250)

    # Models
    embedding_model: str = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_max_tokens: int = 512  # longer text is cut off before embedding
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    rerank: bool = _env("RERANK", False)
    reranker_model: str = _env("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

    # Retrieval
    candidate_k: int = _env("CANDIDATE_K", 20)  # per retriever, before fusion
    top_k: int = _env("TOP_K", 5)  # chunks passed to the LLM
    rrf_k: int = _env("RRF_K", 60)
    # Grounding gate: best dense cosine similarity must reach this, else refuse without calling the LLM.
    # Deliberately loose: it filters clearly unrelated questions; near-topic ones (e.g. Dijkstra)
    # are refused by the LLM's grounding prompt. Tune with eval/run_eval.py on your notes.
    min_similarity: float = _env("MIN_SIMILARITY", 0.45)

    # LLM
    llm_provider: str = _env("LLM_PROVIDER", "groq")  # "groq" | "gemini"
    groq_model: str = _env("GROQ_MODEL", "qwen/qwen3.8-27b")
    gemini_model: str = _env("GEMINI_MODEL", "gemini-2.0-flash")
    temperature: float = _env("TEMPERATURE", 0.1)
    max_tokens: int = _env("MAX_TOKENS", 1024)
    rewrite_followups: bool = _env("REWRITE_FOLLOWUPS", True)

    supported_extensions: tuple = field(default=(".pdf", ".md", ".markdown", ".txt", ".docx"))


settings = Settings()
