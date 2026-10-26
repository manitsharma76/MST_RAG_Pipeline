"""Gradio front end. Hugging Face Spaces runs this file."""
import logging
import os

import gradio as gr

from config import settings
from rag.pipeline import cited_indices, create_pipeline
from rag.prompts import citation_label

logging.basicConfig(level=logging.INFO)

# Loaded once at startup and shared by every request.
pipeline = create_pipeline(settings)

# Render LaTeX from the notes, e.g. $\sum_{e \in T} w(e)$, in answers and passages.
LATEX = [{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}]

EXAMPLES = [
    "What is a minimum spanning tree?",
    "Compare Kruskal's and Prim's algorithms.",
    "Why does the greedy-choice property make greedy MST algorithms correct?",
    "What is the time complexity of Prim's algorithm with a binary heap?",
    "How does Kruskal's algorithm use union-find?",
]


def status_message() -> str:
    problems = []
    if pipeline.retriever.chunks == []:
        problems.append("The knowledge base is empty: add notes to `data/raw/` and run `python scripts/build_index.py`.")
    if pipeline.llm is None:
        problems.append(f"No API key for `{settings.llm_provider}`, so answers are disabled. Add it under Settings → Secrets.")
    return "\n\n".join(f"⚠️ {p}" for p in problems)


def excerpt(text: str, limit: int = 900) -> str:
    """Cut at a line boundary and close an open code block, so the rest of the panel renders."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit("\n", 1)[0]
    if cut.count("```") % 2:
        cut += "\n```"
    return cut + "\n…"


def render_sources(answer: str, chunks, show: bool) -> str:
    if not show or not chunks:
        return ""
    cited = cited_indices(answer)
    blocks = []
    for n, item in enumerate(chunks, start=1):
        match = f"similarity {item.similarity:.2f}" if item.similarity is not None else "keyword match"
        tag = " · **cited**" if n in cited else ""
        text = excerpt(item.chunk.text)
        quoted = "\n".join(f"> {line}" for line in text.splitlines())
        blocks.append(f"**[{n}] {citation_label(item)}** · {match}{tag}\n\n{quoted}")
    return "\n\n---\n\n".join(blocks)


def plain_history(history: list[dict]) -> list[dict]:
    """Gradio 6 sends message content as a list of parts; the pipeline wants plain strings."""
    def text(content) -> str:
        if isinstance(content, str):
            return content
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))

    return [{"role": m["role"], "content": text(m["content"])} for m in history]


def respond(message: str, history: list[dict], top_k: int, show_sources: bool):
    message = message.strip()
    if not message:
        yield "", history, gr.update()
        return
    prior = plain_history(history)
    history = prior + [{"role": "user", "content": message}, {"role": "assistant", "content": "Searching the notes…"}]
    yield "", history, ""
    for answer, chunks in pipeline.stream_answer(message, prior, top_k=int(top_k)):
        history[-1] = {"role": "assistant", "content": answer}
        yield "", history, render_sources(answer, chunks, show_sources)


with gr.Blocks(title="MST Notes Assistant") as demo:
    gr.Markdown(
        "# 🌲 MST Notes Assistant\n"
        "Ask about minimum spanning trees: Prim's and Kruskal's algorithms, the optimal substructure and "
        "greedy-choice properties, correctness proofs and runtimes. Answers come **only from the course "
        "notes** and cite the passages they use."
    )
    status = status_message()
    if status:
        gr.Markdown(status)

    with gr.Sidebar(open=False):
        gr.Markdown("### Settings")
        top_k = gr.Slider(1, 10, value=settings.top_k, step=1, label="Passages to retrieve")
        show_sources = gr.Checkbox(value=True, label="Show retrieved passages")
        gr.Markdown(f"LLM: `{settings.llm_provider}` · Embeddings: `{settings.embedding_model}`")

    chatbot = gr.Chatbot(height=480, buttons=["copy", "copy_all"], label="Chat", latex_delimiters=LATEX)
    with gr.Row():
        question = gr.Textbox(placeholder="Ask a question about minimum spanning trees…", show_label=False, scale=8)
        send = gr.Button("Ask", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=question)
    clear = gr.Button("Clear chat", size="sm")

    with gr.Accordion("Retrieved passages", open=False):
        sources = gr.Markdown(latex_delimiters=LATEX)

    inputs = [question, chatbot, top_k, show_sources]
    outputs = [question, chatbot, sources]
    question.submit(respond, inputs, outputs)
    send.click(respond, inputs, outputs)
    clear.click(lambda: ([], ""), None, [chatbot, sources])


if __name__ == "__main__":
    port = os.getenv("PORT")
    if port:  # hosting platforms route traffic to $PORT on all interfaces
        demo.queue().launch(server_name="0.0.0.0", server_port=int(port))
    else:
        demo.queue().launch()
