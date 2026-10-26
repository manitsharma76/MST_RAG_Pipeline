"""Compare chunk sizes on your notes using eval/questions.jsonl. No LLM calls.

    python scripts/compare_chunking.py
    python scripts/compare_chunking.py --configs 2400:400,1600:250,1000:150

Each size:overlap pair is indexed into its own in-memory Chroma collection, so the real
index in index/ is not touched. For each configuration it reports:
  - over limit:  chunks longer than the embedding model's token limit (their tails are not embedded)
  - hit rate:    in-scope questions whose expected keywords appear in the top-k passages
  - in / out sim: mean best similarity for in-scope and out-of-scope questions
  - gap:         lowest in-scope similarity minus highest out-of-scope similarity; positive means
                 one MIN_SIMILARITY threshold can separate them, and the value is the headroom
"""
import argparse
import sys
import uuid
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb  # noqa: E402

from config import settings  # noqa: E402
from rag.chunking import chunk_documents  # noqa: E402
from rag.embeddings import FastEmbedEmbedder  # noqa: E402
from rag.evaluation import check_retrieval, load_cases  # noqa: E402
from rag.indexing import embed_chunks, load_notes  # noqa: E402
from rag.retriever import HybridRetriever  # noqa: E402
from rag.vectorstore import VectorStore  # noqa: E402


def parse_configs(text: str) -> list[tuple[int, int]]:
    pairs = [part.split(":") for part in text.split(",")]
    return [(int(size), int(overlap)) for size, overlap in pairs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", default=f"2400:400,{settings.chunk_size}:{settings.chunk_overlap},1000:150",
                        help="comma-separated size:overlap pairs, in characters")
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_dir)
    parser.add_argument("--parsed-dir", type=Path, default=settings.parsed_dir)
    parser.add_argument("--questions", type=Path, default=settings.raw_dir.parents[1] / "eval" / "questions.jsonl")
    args = parser.parse_args()

    notes = load_notes(args.raw_dir, args.parsed_dir)
    if not notes:
        sys.exit(f"No notes found in {args.raw_dir}")
    cases = load_cases(args.questions)
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.query_instruction, settings.model_cache_dir)
    token_limit = settings.embedding_max_tokens
    client = chromadb.EphemeralClient()

    summary, per_question = [], {}
    for size, overlap in parse_configs(args.configs):
        label = f"{size}:{overlap}"
        by_source = {source: chunk_documents(docs, size, overlap) for source, docs in notes}
        chunks = [c for source_chunks in by_source.values() for c in source_chunks]
        tokens = [embedder.count_tokens(c.text) for c in chunks]

        store = VectorStore(f"compare_{uuid.uuid4().hex[:8]}", client=client)
        for source, source_chunks in by_source.items():
            store.replace_source(source, source_chunks, embed_chunks(embedder, source_chunks))
        retriever = HybridRetriever(store, embedder, settings.candidate_k, settings.rrf_k)

        results = [check_retrieval(retriever, case, settings.top_k, settings.min_similarity) for case in cases]
        inside = [r.retrieval.best_similarity for r in results if r.case["in_scope"]]
        outside = [r.retrieval.best_similarity for r in results if not r.case["in_scope"]]
        for r in results:
            per_question.setdefault(r.case["question"], {})[label] = r

        summary.append({
            "config": label,
            "chunks": len(chunks),
            "avg tokens": round(mean(tokens)),
            "max tokens": max(tokens),
            "over limit": sum(t > token_limit for t in tokens),
            "hit rate": f"{sum(r.hit for r in results)}/{len(inside)}",
            "in sim": f"{mean(inside):.3f}" if inside else "-",
            "out sim": f"{mean(outside):.3f}" if outside else "-",
            "gap": f"{min(inside) - max(outside):+.3f}" if inside and outside else "-",
        })

    print(f"\nEmbedding model: {settings.embedding_model} (token limit {token_limit}), "
          f"top_k={settings.top_k}, MIN_SIMILARITY={settings.min_similarity}\n")
    print_table(summary)

    labels = [row["config"] for row in summary]
    print("\nBest similarity per question (* = retrieval hit, ✗ = wrong side of the gate):\n")
    rows = []
    for question, by_label in per_question.items():
        scope = "in" if next(iter(by_label.values())).case["in_scope"] else "out"
        row = {"scope": scope, "question": question[:55]}
        for label in labels:
            r = by_label[label]
            row[label] = f"{r.retrieval.best_similarity:.3f}{'*' if r.hit else ' '}{' ' if r.ok else '✗'}"
        rows.append(row)
    print_table(rows)


def print_table(rows: list[dict]) -> None:
    headers = list(rows[0])
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for r in rows:
        print("  ".join(str(r[h]).ljust(widths[h]) for h in headers))


if __name__ == "__main__":
    main()
