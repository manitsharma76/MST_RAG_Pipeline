import re

from rag.retriever import RetrievedChunk

NOT_COVERED = (
    "I couldn't find this in the notes, so I can't answer it from the knowledge base. "
    "Try rephrasing, or ask about a minimum spanning tree topic the notes cover "
    "(for example Prim's or Kruskal's algorithm, the greedy-choice property, or their runtimes)."
)

SYSTEM_PROMPT = f"""You are a teaching assistant for a Data Structures and Algorithms course. \
You answer questions about minimum spanning trees using ONLY the numbered context passages \
taken from the course notes.

Rules:
1. Use only facts stated in the context. Do not add facts, examples, proofs, complexities or \
code from your own knowledge, even if you are sure they are correct.
2. Cite every factual sentence with the passage number(s) it comes from, like [1] or [2][3].
3. If the context only partly answers the question, answer that part and say clearly which \
part the notes do not cover.
4. If the context does not answer the question at all, reply exactly: "{NOT_COVERED}"
5. Keep the notes' notation and terminology. Copy pseudocode, formulas and table values exactly \
as they appear in the context: keep symbols such as ≠, ∉ and ∞, keep LaTeX math in $...$ or \
$$...$$, and never "correct" or simplify them. Put pseudocode in a code block with its line \
numbers and indentation. Otherwise use Markdown: short paragraphs and lists for steps."""

REWRITE_PROMPT = """Rewrite the user's latest question so it can be understood without the \
conversation, by replacing pronouns and references ("it", "that algorithm", "the second one") \
with what they refer to. Do not answer the question. If it is already self-contained, return \
it unchanged. Reply with the rewritten question only."""


def display_source(source: str) -> str:
    """Drop a download hash prefix: "4a7f…63a_MIT6_046JS15_lec12.pdf" -> "MIT6_046JS15_lec12.pdf"."""
    return re.sub(r"^[0-9a-f]{32}_", "", source)


def citation_label(chunk: RetrievedChunk) -> str:
    meta = chunk.chunk.metadata
    parts = [display_source(meta["source"])]
    if meta.get("page"):
        end = meta.get("page_end")
        parts.append(f"p. {meta['page']}" + (f"–{end}" if end and end != meta["page"] else ""))
    if meta.get("section"):
        parts.append(meta["section"])
    return ", ".join(parts)


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{n}] ({citation_label(c)})\n{c.chunk.text}" for n, c in enumerate(chunks, start=1))


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    return f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {question}"


def build_rewrite_message(question: str, history: list[dict]) -> str:
    transcript = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)
    return f"Conversation:\n{transcript}\n\nLatest question: {question}"
