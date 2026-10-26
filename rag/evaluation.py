"""Retrieval checks shared by eval/run_eval.py and scripts/compare_chunking.py."""
import json
from dataclasses import dataclass
from pathlib import Path

from rag.retriever import HybridRetriever, RetrievalResult


@dataclass
class CaseResult:
    case: dict
    retrieval: RetrievalResult
    hit: bool  # in-scope: an expected keyword appears in the retrieved passages
    passes_gate: bool  # best similarity reached MIN_SIMILARITY

    @property
    def ok(self) -> bool:
        """In-scope: found and not refused. Out-of-scope: refused by the gate."""
        return self.hit and self.passes_gate if self.case["in_scope"] else not self.passes_gate


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_retrieval(retriever: HybridRetriever, case: dict, top_k: int, min_similarity: float) -> CaseResult:
    result = retriever.retrieve(case["question"], top_k=top_k)
    # Collapse whitespace so a keyword phrase still matches across PDF line breaks.
    text = " ".join(" ".join(c.chunk.text.lower().split()) for c in result.chunks)
    hit = case["in_scope"] and any(k.lower() in text for k in case["expected_keywords"])
    return CaseResult(case, result, hit, result.best_similarity >= min_similarity)
