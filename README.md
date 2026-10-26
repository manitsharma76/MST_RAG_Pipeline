---
title: MST Notes Assistant
emoji: 🌲
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# MST Notes Assistant

A retrieval-augmented question-answering app over MIT 6.046J Lecture 12 notes on **minimum spanning trees** (Prim's and Kruskal's algorithms, the greedy-choice and optimal substructure properties, correctness proofs and runtimes). Every answer is grounded in the notes, with page-level citations. If the notes don't cover a question, the app says so.

## How it works

1. **Ingest** (`scripts/build_index.py`): load files from `data/raw/` (PDF, Markdown, TXT, DOCX), clean them, split them into heading-aware chunks, embed them with `BAAI/bge-small-en-v1.5` (run with fastembed on ONNX Runtime, no PyTorch), and store them in an embedded ChromaDB under `index/chroma/`.
2. **Retrieve**: run BM25 keyword search and dense vector search, fuse the results with Reciprocal Rank Fusion, and optionally rerank with a cross-encoder.
3. **Grounding gate**: if no chunk is similar enough to the question, refuse without calling the LLM.
4. **Generate**: the LLM (Groq or Gemini, both free tier) answers only from the retrieved chunks and cites them as `[1]`, `[2]`, and so on.

## Preparing PDF notes

Plain text extraction damages math-heavy PDFs. For example, `≠` comes out as `=`, Σ and superscripts are lost, and tables come out column by column. Convert each PDF with LlamaParse once, then review the result:

```bash
python scripts/prepare_notes.py      # data/raw/<name>.pdf -> data/parsed/<name>.md (needs LLAMA_CLOUD_API_KEY)
```

Check `data/parsed/<name>.md` against the PDF, especially the pseudocode, tables and `$math$`, and fix anything that differs. Then commit it. The indexer uses a reviewed Markdown file in place of the PDF with the same name, so the Space never needs a LlamaParse key. PDFs without a reviewed file fall back to direct text extraction.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then add your GROQ_API_KEY
# put your notes in data/raw/
python scripts/build_index.py   # build the index
python app.py                   # http://localhost:7860
```

Run the tests with `pytest`. Run the evaluation with `python eval/run_eval.py`.
To compare chunk sizes on your notes, run `python scripts/compare_chunking.py`. It uses temporary in-memory indexes and makes no LLM calls.

## Deployment

```
Hugging Face Static Space (free)            Render free web service
frontend/index.html + config.js   ──────►   api.py (FastAPI): retrieval + Groq
  chat UI, Markdown/LaTeX, sources   POST    GROQ_API_KEY kept server-side
                                  /api/ask
```

The Groq key can't go in the static page, because anyone can read a static page's source. So the page calls the backend, which holds the key. The backend only accepts browser requests from `ALLOWED_ORIGINS`, and it limits each IP address to `RATE_LIMIT_PER_MINUTE` questions.

1. **Backend (Render).** Push this repo to GitHub. In Render, choose **New → Blueprint**, pick the repo, and enter `GROQ_API_KEY` when asked. `render.yaml` installs `requirements-backend.txt`, then builds the index and caches the model during the build. The free instance sleeps after 15 idle minutes, and the first request after that takes about a minute.
2. **Frontend (Hugging Face).** Run `python scripts/deploy_frontend.py --space <user>/<space> --api-url https://<service>.onrender.com`. Then make sure the printed `*.static.hf.space` origin is in the backend's `ALLOWED_ORIGINS`.

To run the API locally, use `uvicorn api:app --port 8000` together with `python -m http.server 8080 -d frontend`. `frontend/config.js` points at port 8000.

The Gradio app (`app.py`) is still the simplest way to run everything locally. On Hugging Face it needs a Gradio Space, which is either a PRO account on CPU Basic, or a ZeroGPU Space for free accounts at least 30 days old (`scripts/deploy_space.py`).
