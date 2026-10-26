# MST Notes Assistant

A retrieval-augmented generation (RAG) app that answers questions about **minimum spanning trees**, using only a set of course notes: the MIT 6.046J Lecture 12 notes, *Greedy Algorithms and Minimum Spanning Tree*. Every answer is built from passages retrieved from the notes and cites them by page and section. If the notes don't cover a question, the app says so instead of answering from the model's general knowledge.

- **Interface:** a static Hugging Face Space at [huggingface.co/spaces/Mshxrma/Test-Retrieval-Pipeline](https://huggingface.co/spaces/Mshxrma/Test-Retrieval-Pipeline)
- **Backend:** a FastAPI service on Render at `https://mst-rag-api.onrender.com`
- **Cost:** every component runs on a free tier.

---

## Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [How the pipeline works](#how-the-pipeline-works)
4. [Evaluation](#evaluation)
5. [Project structure](#project-structure)
6. [Configuration](#configuration)
7. [Running locally](#running-locally)
8. [Deployment](#deployment)
9. [Testing](#testing)
10. [Design decisions](#design-decisions)
11. [Known limitations](#known-limitations)

---

## Features

- **Grounded answers with citations.** The model may only use the retrieved passages. It cites them as `[1]`, `[2]`, …, and each citation maps to a source, page and section.
- **Two layers of refusal.** A similarity threshold rejects clearly unrelated questions without calling the LLM. The LLM itself is instructed to refuse topics the passages don't cover, such as Borůvka's algorithm or Dijkstra's algorithm.
- **Accurate pseudocode and math.** The PDF is converted with LlamaParse and the result is reviewed by hand. This preserves `≠`, `Σ`, superscripts, subscripts, tables, line numbers and indentation.
- **Hybrid retrieval.** BM25 keyword search and dense vector search are fused with Reciprocal Rank Fusion. An optional cross-encoder reranker is available.
- **Math-aware search.** Symbols are spelled out for search only (`≠` becomes "not equal", `\sum` becomes "sum", `lg` also matches "log"). The model and the UI still see the exact notation.
- **Follow-up questions.** "What about its runtime?" is rewritten into a standalone question before retrieval.
- **Resilient generation on free tiers.** The app tries Groq (`qwen/qwen3.8-27b`) first, then Groq's `openai/gpt-oss-120b`, then Gemini (`gemini-3.5-flash-lite`). A rate-limited provider is skipped for 60 seconds.
- **Streaming UI.** Answers stream in as they're generated. Markdown and LaTeX are rendered with KaTeX, and a sources panel marks the cited passages.
- **Measurable.** An evaluation set and scripts compare chunk sizes, PDF parsers and models.

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
  Browser                │  Hugging Face Static Space (free)            │
  ───────                │  frontend/index.html                         │
  question ─────────────►│  chat UI · Markdown + KaTeX · sources panel  │
                         └───────────────────┬──────────────────────────┘
                                             │ POST /api/ask  (streamed NDJSON)
                                             ▼
                         ┌──────────────────────────────────────────────┐
                         │  Render web service (free) · api.py          │
                         │  CORS allow-list · 10 questions/min per IP   │
                         │                                              │
                         │  1. rewrite follow-up → standalone question  │
                         │  2. hybrid retrieval  (BM25 + dense, RRF)    │
                         │  3. grounding gate    (min similarity)       │
                         │  4. generate + cite   (LLM provider chain)   │
                         └──────┬───────────────────────────┬───────────┘
                                │                           │
                 ┌──────────────▼─────────────┐   ┌─────────▼────────────────────┐
                 │ ChromaDB (embedded, local) │   │ Groq: qwen3.8-27b             │
                 │ fastembed bge-small (ONNX) │   │   → gpt-oss-120b (fallback)   │
                 │ built at deploy time       │   │ Gemini 3.5 Flash-Lite (backup)│
                 └────────────────────────────┘   └──────────────────────────────┘
```

**Why split the frontend and the backend?** The project must be hosted on Hugging Face. However, Hugging Face now requires a paid plan for Gradio and Docker Spaces on CPU hardware, while static Spaces stay free. A static page can't hold an API key, because anyone can read its source. So the page runs on Hugging Face, and the retrieval and LLM calls run on a free Render service that keeps the keys server-side.

**Offline vs. online.** Building the knowledge base (parsing, chunking, embedding) happens ahead of time, during Render's build step. Answering a question only loads the stored index and runs the four steps shown above.

---

## How the pipeline works

### 1. Preparing the notes: `scripts/prepare_notes.py`

Math-heavy PDFs produced by LaTeX don't survive plain text extraction. I compared three parsers on the lecture notes using `scripts/compare_parsers.py`:

| Checkpoint | PyMuPDF | pdfplumber | **LlamaParse (agentic)** |
|---|---|---|---|
| Kruskal's condition `Find-Set(u) ≠ Find-Set(v)` | ✗ extracted as `=` (inverted) | ✗ extracted as `=` | ✓ |
| Prim's pseudocode, all 12 lines with indentation | ✗ 6 of 12 lines | ✓ | ✓ |
| MST objective `Σ_{e∈T} w(e)` | ✗ Σ lost | ✗ Σ lost | ✓ LaTeX |
| Superscripts and subscripts (`O(V²)`, `T_{Extract-Min}`) | ✗ | ✗ | ✓ |
| Runtime table | ✗ read column by column | ~ aligned text | ✓ real table |

The `≠` failure comes from the PDF itself. The PDF draws "≠" as an "=" with a separate slash glyph on top, and the slash's font maps it to a space character. Any text-only parser therefore reports the opposite condition.

**The workflow:**
1. `prepare_notes.py` sends each PDF to LlamaParse's v2 API. Responses are cached under `parsed/`, so re-runs cost no credits.
2. It cleans the result: removes `**bold**` markers, and converts HTML tables to Markdown tables with `^` and `_` for superscripts and subscripts.
3. It writes `data/parsed/<name>.md`, with `<!-- page N -->` markers so citations keep their page numbers.
4. A person reviews that file against the PDF and commits it. LlamaParse reads page images with a model and can occasionally "correct" the source.

The indexer uses a reviewed Markdown file in place of the PDF with the same name. A PDF without a reviewed file falls back to direct PyMuPDF extraction, and the build prints a warning.

### 2. Loading and chunking: `rag/loaders/`, `rag/chunking.py`

- **Loaders.** A registry maps file extensions to loaders: PDF, Markdown/TXT and DOCX. Every loader produces Markdown-style text with `#` headings, so the chunker treats all formats the same way.
  - The PDF loader detects headings from font size and boldness.
  - It removes running headers and footers that repeat on most pages, and joins words hyphenated across lines, including soft hyphens.
- **Heading-aware chunking.** Text is split into sections at headings, and sections are split into paragraphs. Paragraphs are packed into chunks of at most **1,600 characters** (about 400 tokens), with about **250 characters of overlap**.
  - **Code blocks are never split at blank lines.** If one is too large for a single chunk, each piece is re-fenced.
  - **Large tables repeat their header row** in every piece.
  - **Every chunk starts with its section path,** e.g. `Prim's Algorithm > Runtime`. This helps both retrieval and citations.
  - **Generic headings nest under their topic.** "Correctness", "Runtime", "Pseudocode" and similar headings are placed under the nearest topic heading. This keeps Prim's and Kruskal's runtime sections distinct even when the notes use the same heading level for all of them.
- **Chunk size choice.** `scripts/compare_chunking.py` indexes the notes at several sizes and compares them. 1,600 characters keeps every chunk under the embedding model's 512-token limit (the largest is 442 tokens) and retrieves just as well as smaller sizes.

### 3. Indexing: `rag/indexing.py`, `rag/embeddings.py`, `rag/vectorstore.py`

- **Embeddings:** `BAAI/bge-small-en-v1.5`, run with **fastembed** on ONNX Runtime, with no PyTorch.
  - The vectors match sentence-transformers': I measured a cosine similarity of 1.0000 between them.
  - Peak memory drops from about 660 MB to about 460 MB, which lets the backend fit Render's free 512 MB instance.
- **Search text.** `rag/search_text.py` creates a search-only version of each chunk, with symbols and LaTeX spelled out as words. Both the embeddings and BM25 are built from this version. The stored chunk keeps the exact notation.
- **Vector store:** ChromaDB in embedded mode, persisted to `index/chroma/` with cosine distance.
  - Each chunk carries metadata: source, page range, section, the algorithms it mentions, and its position.
  - Re-indexing a source replaces its chunks, so files can be added or edited without duplicates.

### 4. Retrieval: `rag/retriever.py`

1. **Dense search:** the question gets BGE's query instruction prefix, and Chroma returns the top 20 chunks by cosine similarity.
2. **Keyword search:** BM25 (`rank_bm25`) also returns its top 20. BM25 catches exact terms such as `Find-Set`, `Borůvka` and `O(E lg V)` that embeddings can blur.
3. **Fusion:** Reciprocal Rank Fusion combines the two lists, scoring each chunk as `Σ 1/(60 + rank)`. It only uses rank positions, so the two methods' different score scales never need to be compared.
4. **Optional reranking:** setting `RERANK=true` adds a cross-encoder (`ms-marco-MiniLM-L-6-v2`, ONNX), which re-scores the candidates.
5. **Result:** the top 5 chunks go to the LLM.

**Grounding gate.** If the best dense similarity is below `MIN_SIMILARITY` (0.45), the app refuses without calling the LLM. The threshold is deliberately loose. Near-topic questions like Borůvka score as high as real ones (about 0.77), so those are left to the LLM's own refusal.

### 5. Generation: `rag/pipeline.py`, `rag/prompts.py`, `rag/llm/`

- **Follow-up rewriting.** When there is chat history, the LLM rewrites the latest question so it stands alone; for example, "its runtime" becomes "the runtime of Prim's algorithm". This call reserves at most 150 tokens.
- **The prompt contains only the question and the numbered passages,** never the chat history. That way an earlier answer can't be repeated as if it were a fact from the notes.
- **The system prompt's rules:**
  - Use only the passages, and cite every factual sentence.
  - If the passages don't answer the question, reply with the exact refusal text.
  - Copy pseudocode, formulas and table values exactly, keeping the LaTeX.
  - Keep answers to about 250 words.
- **The provider chain** (`rag/llm/fallback.py`):
  - **The order:** Groq `qwen/qwen3.8-27b`, then Groq `openai/gpt-oss-120b` (with `reasoning_effort=low`), then Gemini `gemini-3.5-flash-lite`.
  - **Rate limits:** a rate-limited provider is skipped for 60 seconds.
  - **Other errors:** an error such as "model overloaded" or "model retired" also moves the request to the next provider, but only before any text has been streamed.
- **Free-tier limits.** Groq's free-tier limit on output tokens per minute changes with load (it has been as low as 1,000). So answers reserve at most 700 tokens.
  - If an answer still hits the cap, it ends with a note suggesting the user ask about one part at a time.
  - If every provider is busy, visitors see a short "try again in a minute" message. The full error goes only to the server logs.

### 6. Interfaces

- **`frontend/index.html` (deployed):** a single self-contained page.
  - It streams NDJSON events from `/api/ask` and renders Markdown with marked and DOMPurify, and math with KaTeX.
  - It shows the retrieved passages with similarity scores and marks which ones were cited.
  - When the Render instance is asleep, it shows "Waking up the server".
  - The deploy script writes the backend URL into the page itself. A private Space signs only the page request, so a separate `config.js` file would be blocked.
- **`api.py` (deployed):** a FastAPI app.
  - It restricts cross-origin requests to an allow-list, limits each IP to a number of questions per minute, and caps the size of questions and history.
  - Its `/api/health` endpoint reports the number of indexed chunks and the active model chain.
- **`app.py` (local):** a Gradio chat app with the same pipeline. It's the quickest way to try the project locally.

---

## Evaluation

`eval/questions.jsonl` holds 30 questions about the lecture:
- **22 in-scope questions.** Five of them depend on pseudocode, formulas or the runtime table.
- **8 out-of-scope questions.** Five are near misses about graphs or MSTs that the notes don't cover (Borůvka, reverse-delete, the cycle property, union-by-rank, Dijkstra). Three are off-topic (quicksort, a linked list, the capital of France).

```bash
python eval/run_eval.py                    # retrieval and grounding gate only (no LLM calls)
python eval/run_eval.py --llm --delay 5    # also check citations and refusals in real answers
```

| Metric | Result |
|---|---|
| Retrieval hit rate: the expected passage is in the top 5 | **22 / 22** |
| In-scope answers that cite the notes (`qwen/qwen3.8-27b`) | **22 / 22** |
| Out-of-scope questions refused | **8 / 8** |
| Grounding gate alone (without the LLM) | 23 / 30: the gate catches only clearly unrelated questions; the LLM refuses near misses |

**Chunk-size comparison** (`scripts/compare_chunking.py`):

| Size : overlap (characters) | Chunks | Max tokens | Over the 512-token limit | Hit rate |
|---|---|---|---|---|
| 2400 : 400 | 16 | 442 | 0 | 22/22 |
| **1600 : 250** (default) | 16 | 442 | 0 | 22/22 |
| 1000 : 150 | 17 | 353 | 0 | 22/22 |
| 600 : 100 | 23 | 217 | 0 | 22/22 |

The sections in these notes are short, so heading-based splitting decides most chunk boundaries. 1,600 characters leaves room for longer notes while staying under the token limit.

---

## Project structure

```
├── api.py                      # FastAPI backend (deployed on Render)
├── app.py                      # Gradio app (local use / Gradio Space)
├── config.py                   # all settings, each overridable by an environment variable
├── render.yaml                 # Render Blueprint: build index at deploy time, start uvicorn
├── requirements.txt            # full local environment (Gradio app, tests)
├── requirements-backend.txt    # backend only, no Gradio (smaller, less memory)
├── .python-version             # Python 3.12 for Render
├── data/
│   ├── raw/                    # source PDFs
│   └── parsed/                 # reviewed Markdown from LlamaParse (what gets indexed)
├── frontend/
│   ├── index.html              # static chat page (deployed to the HF Space)
│   └── config.js               # backend URL for local testing
├── rag/
│   ├── loaders/                # PDF / Markdown / DOCX loaders + registry
│   ├── parsing/                # LlamaParse v2 client, Markdown cleanup
│   ├── preprocessing.py        # unicode, hyphenation, page numbers, running headers
│   ├── chunking.py             # heading-aware chunking (code/table aware)
│   ├── search_text.py          # symbols → words for embeddings and BM25
│   ├── embeddings.py           # fastembed embedder and reranker
│   ├── vectorstore.py          # embedded ChromaDB wrapper
│   ├── indexing.py             # choose note files, chunk, embed, store
│   ├── retriever.py            # BM25 + dense + RRF (+ rerank)
│   ├── prompts.py              # system prompt, citations, refusal text
│   ├── pipeline.py             # rewrite → retrieve → gate → generate
│   ├── evaluation.py           # retrieval checks shared by the eval scripts
│   └── llm/                    # Groq and Gemini clients, fallback chain
├── scripts/
│   ├── prepare_notes.py        # PDF → reviewed Markdown via LlamaParse
│   ├── build_index.py          # build or rebuild the Chroma index
│   ├── compare_parsers.py      # PyMuPDF vs pdfplumber vs LlamaParse
│   ├── compare_chunking.py     # chunk-size comparison on the eval set
│   ├── deploy_frontend.py      # publish frontend/ to a static HF Space
│   └── deploy_space.py         # publish the Gradio app (needs HF PRO or ZeroGPU)
├── eval/
│   ├── questions.jsonl         # 30 in-scope and out-of-scope questions
│   └── run_eval.py             # retrieval / citation / refusal evaluation
└── tests/                      # 39 offline tests (fake embedder and LLM)
```

---

## Configuration

All settings are in `config.py`. Locally they're read from `.env` (copy `.env.example`); on Render, from the service's environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq key (`gsk_…`) |
| `GEMINI_API_KEY` | — | Optional backup provider, used when Groq is rate-limited |
| `LLM_PROVIDER` | `groq` | Main provider (`groq` or `gemini`) |
| `LLM_BACKUP_PROVIDERS` | `gemini` | Backups, in order; each is used only if its key is set |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Main Groq model |
| `GROQ_FALLBACK_MODELS` | `openai/gpt-oss-120b` | Tried when the main Groq model is rate-limited |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini model |
| `MAX_TOKENS` / `REWRITE_MAX_TOKENS` | `700` / `150` | Output tokens reserved for answers / follow-up rewrites |
| `TEMPERATURE` | `0.1` | Generation temperature |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1600` / `250` | Chunking, in characters |
| `TOP_K` / `CANDIDATE_K` / `RRF_K` | `5` / `20` / `60` | Passages sent to the LLM / candidates per retriever / RRF constant |
| `MIN_SIMILARITY` | `0.45` | Grounding gate threshold |
| `RERANK` | `false` | Enable the cross-encoder reranker |
| `REWRITE_FOLLOWUPS` | `true` | Rewrite follow-up questions using the chat history |
| `ALLOWED_ORIGINS` | localhost:8080 | Backend CORS allow-list (comma-separated) |
| `RATE_LIMIT_PER_MINUTE` | `10` | Questions per minute per client IP |
| `LLAMA_CLOUD_API_KEY` | — | Only for `prepare_notes.py` and `compare_parsers.py` |

---

## Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                    # add GROQ_API_KEY (and optionally GEMINI_API_KEY)

python scripts/build_index.py --rebuild # index data/parsed/ (and any PDFs without a reviewed file)
python app.py                           # Gradio app at http://localhost:7860
```

To run the deployed setup locally (API + static page):

```bash
uvicorn api:app --port 8000
python -m http.server 8080 -d frontend  # open http://localhost:8080
```

**Adding notes:**
1. Put the PDF in `data/raw/`.
2. Run `python scripts/prepare_notes.py`. This needs `LLAMA_CLOUD_API_KEY`.
3. Review `data/parsed/<name>.md` against the PDF.
4. Rebuild the index with `python scripts/build_index.py --rebuild`.
5. Add questions about the new material to `eval/questions.jsonl`.

---

## Deployment

| Part | Host | How it updates |
|---|---|---|
| Backend (`api.py`) | Render, free web service, from `render.yaml` | **Automatically** on every push to `main`. The build installs `requirements-backend.txt`, downloads the model and rebuilds the index. |
| Frontend (`frontend/index.html`) | Hugging Face static Space | **Manually:** `python scripts/deploy_frontend.py --space Mshxrma/Test-Retrieval-Pipeline --api-url https://mst-rag-api.onrender.com --private` |

**First-time setup:**
1. Push this repository to GitHub.
2. In Render, choose **New → Blueprint** and pick the repo. Enter `GROQ_API_KEY` when asked.
3. Add `GEMINI_API_KEY` under the service's **Environment** settings.
4. Run `deploy_frontend.py` with the Render URL.
5. Make sure the Space's address (`https://<user>-<space>.static.hf.space`) is in the backend's `ALLOWED_ORIGINS`.

**Checking the live backend:** `curl https://mst-rag-api.onrender.com/api/health` returns the chunk count and the active model chain.

---

## Testing

```bash
pytest          # 39 tests, offline, under a second
```

The tests use a deterministic fake embedder, an in-memory Chroma instance, and fake LLM and Groq clients. They cover:
- loading and cleaning, including PDF headings, running headers and soft hyphens;
- LlamaParse Markdown cleanup and page markers;
- chunking: code blocks, tables, overlap, and heading nesting;
- search text, hybrid retrieval and reranking;
- the grounding gate and follow-up rewriting;
- error handling, Groq model fallback, the provider chain and cool-downs.

---

## Design decisions

- **No LangChain.** The hard parts of this project are PDF fidelity, chunking that respects code and math, and free-tier rate limits. A framework doesn't help with any of them. Using the libraries directly keeps every step visible and testable, and avoids frequent breaking API changes.
- **Embedded ChromaDB, not a hosted vector database.** The corpus is small, so a hosted database would add an account, network latency and quota limits for no benefit. Chroma's metadata filtering and replace-by-source updates support adding more notes later.
- **Hybrid retrieval.** Dense search handles paraphrases. BM25 handles exact notation. RRF combines them without tuning weights.
- **The LLM sees no chat history.** History is used only to rewrite the question, so answers can't be built on earlier, unverified answers.
- **The LLM is the second refusal layer.** In-scope and near-topic questions get similar similarity scores, so no single threshold separates them. The threshold only filters obviously unrelated questions.

---

## Known limitations

- **Cold starts:** Render's free instance sleeps after 15 idle minutes, so the first question after that takes about a minute.
- **Free-tier quotas:** Groq allows about 1,000 requests per day per model, plus per-minute token limits that change with load. Gemini's free tier allows only about a dozen requests per minute. Heavy use will hit these limits.
- **Tight memory:** the backend runs close to Render's 512 MB limit. If it is ever killed for using too much memory, limit the ONNX Runtime threads, or move to a paid instance.
- **Reviewed notes are a manual step:** LlamaParse output should be checked against the PDF. For example, the parsed Lemma 1 proof reads `w(T^*)` where the PDF has `w(T')`.
- **The knowledge base is one lecture.** Questions outside it are refused by design.
