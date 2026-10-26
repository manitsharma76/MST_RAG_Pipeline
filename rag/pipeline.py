import logging
import re
from collections.abc import Iterator

from rag.llm import LLMClient
from rag.prompts import NOT_COVERED, REWRITE_PROMPT, SYSTEM_PROMPT, build_rewrite_message, build_user_message
from rag.retriever import HybridRetriever, RetrievedChunk

log = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: LLMClient | None, min_similarity: float,
                 rewrite_followups: bool = True, history_turns: int = 6):
        self.retriever = retriever
        self.llm = llm
        self.min_similarity = min_similarity
        self.rewrite_followups = rewrite_followups
        self.history_turns = history_turns

    def stream_answer(self, question: str, history: list[dict] | None = None,
                      top_k: int = 5) -> Iterator[tuple[str, list[RetrievedChunk]]]:
        """Yield (answer so far, retrieved chunks) as the answer streams in."""
        query = self.standalone_question(question, history or [])
        result = self.retriever.retrieve(query, top_k=top_k)

        # Grounding gate: nothing in the notes is close enough, so don't let the LLM improvise.
        if not result.chunks or result.best_similarity < self.min_similarity:
            yield NOT_COVERED, result.chunks
            return

        if self.llm is None:
            yield "The language model isn't configured (missing API key), so here are the most relevant passages.", result.chunks
            return

        # The LLM gets only the (standalone) question and the passages, not the chat history,
        # so earlier answers can't leak in as ungrounded "facts".
        messages = [{"role": "user", "content": build_user_message(query, result.chunks)}]
        answer = ""
        try:
            for token in self.llm.stream(SYSTEM_PROMPT, messages):
                answer += token
                yield answer, result.chunks
        except Exception as exc:  # rate limits, network errors, bad keys
            log.exception("LLM call failed")
            yield f"{answer}\n\n⚠️ The language model request failed: {exc}", result.chunks

    def answer(self, question: str, history: list[dict] | None = None, top_k: int = 5) -> tuple[str, list[RetrievedChunk]]:
        final = ("", [])
        for final in self.stream_answer(question, history, top_k):
            pass
        return final

    def standalone_question(self, question: str, history: list[dict]) -> str:
        """Rewrite follow-ups such as "what about its complexity?" so retrieval can find them."""
        if not (self.rewrite_followups and history and self.llm):
            return question
        recent = [m for m in history if m.get("role") in {"user", "assistant"}][-self.history_turns:]
        try:
            rewritten = self.llm.generate(REWRITE_PROMPT, [{"role": "user", "content": build_rewrite_message(question, recent)}]).strip()
            return rewritten or question
        except Exception:
            log.exception("Follow-up rewrite failed; using the original question")
            return question


def cited_indices(answer: str) -> set[int]:
    """1-based passage numbers cited in the answer, e.g. "[1][3]" -> {1, 3}."""
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}


def create_pipeline(settings) -> RAGPipeline:
    """Wire up the production pipeline; builds the index on first run if it is missing."""
    from rag.embeddings import FastEmbedEmbedder, FastEmbedReranker
    from rag.indexing import build_index
    from rag.llm import MissingAPIKeyError, get_llm
    from rag.vectorstore import VectorStore

    embedder = FastEmbedEmbedder(settings.embedding_model, settings.query_instruction, settings.model_cache_dir)
    store = VectorStore(settings.collection_name, path=settings.index_dir / "chroma")
    if store.count() == 0 and settings.raw_dir.exists():
        log.info("Index is empty; building it from %s", settings.raw_dir)
        build_index(settings.raw_dir, store, embedder, settings.chunk_size, settings.chunk_overlap,
                    parsed_dir=settings.parsed_dir)

    reranker = FastEmbedReranker(settings.reranker_model, settings.model_cache_dir) if settings.rerank else None

    try:
        llm = get_llm(settings)
    except MissingAPIKeyError as exc:
        log.warning("%s", exc)
        llm = None

    retriever = HybridRetriever(store, embedder, settings.candidate_k, settings.rrf_k, reranker)
    return RAGPipeline(retriever, llm, settings.min_similarity, settings.rewrite_followups)
