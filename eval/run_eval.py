"""Evaluate retrieval and grounding against eval/questions.jsonl.

    python eval/run_eval.py          # retrieval + grounding gate only (no LLM calls)
    python eval/run_eval.py --llm    # also generate answers and check citations / refusals

Edit questions.jsonl so the in-scope questions and keywords match your notes.
A question is a retrieval hit if any expected keyword appears in the retrieved passages.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from rag.evaluation import check_retrieval, load_cases  # noqa: E402
from rag.pipeline import cited_indices, create_pipeline  # noqa: E402
from rag.prompts import NOT_COVERED  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="also call the LLM (uses API quota)")
    parser.add_argument("--file", default=Path(__file__).with_name("questions.jsonl"), type=Path)
    args = parser.parse_args()

    pipeline = create_pipeline(settings)
    cases = load_cases(args.file)

    hits = in_scope = gate_ok = cited_ok = refused_ok = 0
    print(f"{'':2} {'sim':>5}  question")
    for case in cases:
        r = check_retrieval(pipeline.retriever, case, settings.top_k, settings.min_similarity)
        in_scope += case["in_scope"]
        hits += r.hit
        gate_ok += r.passes_gate if case["in_scope"] else not r.passes_gate
        ok = r.ok

        note = ""
        if args.llm:
            answer, _ = pipeline.answer(case["question"])
            refused = NOT_COVERED in answer
            if case["in_scope"]:
                cited = bool(cited_indices(answer)) and not refused
                cited_ok += cited
                ok = ok and cited
                note = "" if cited else "  (no citation or refused)"
            else:
                refused_ok += refused
                ok = refused
                note = "" if refused else "  (answered an out-of-scope question)"

        print(f"{'✓' if ok else '✗':2} {r.retrieval.best_similarity:5.2f}  {case['question']}{note}")

    out_scope = len(cases) - in_scope
    print(f"\nRetrieval hit rate (in-scope):  {hits}/{in_scope}")
    print(f"Grounding gate correct:          {gate_ok}/{len(cases)}  (threshold MIN_SIMILARITY={settings.min_similarity})")
    if args.llm:
        print(f"Cited answers (in-scope):        {cited_ok}/{in_scope}")
        print(f"Refusals (out-of-scope):         {refused_ok}/{out_scope}")
    print("\nTune MIN_SIMILARITY so in-scope questions sit above it and out-of-scope ones below it.")


if __name__ == "__main__":
    main()
