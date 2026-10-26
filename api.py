"""HTTP backend for the static Hugging Face Space frontend (frontend/index.html).

    uvicorn api:app --port 8000

POST /api/ask streams newline-delimited JSON events:
    {"type": "sources", "sources": [...]}        retrieved passages (sent first)
    {"type": "delta", "text": "..."}             the next piece of the answer
    {"type": "done", "answer": "...", "cited": [1, 3]}
    {"type": "error", "message": "..."}
GET /api/health reports whether the index and LLM are ready (the frontend uses it to wake the server).

The Groq key stays on this server. CORS only admits ALLOWED_ORIGINS, and each client IP is
limited to RATE_LIMIT_PER_MINUTE questions, so the public endpoint can't drain the quota.
"""
import json
import logging
import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from rag.pipeline import cited_indices, create_pipeline
from rag.prompts import citation_label

logging.basicConfig(level=logging.INFO)

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080").split(",") if o.strip()]
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
MAX_QUESTION_CHARS = 1000
MAX_HISTORY_MESSAGES = 6

# Loaded once at startup and shared by every request.
pipeline = create_pipeline(settings)

app = FastAPI(title="MST Notes Assistant API", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[Message] = Field(default_factory=list)
    top_k: int = Field(default=settings.top_k, ge=1, le=10)


_requests: dict[str, deque] = defaultdict(deque)


def check_rate_limit(request: Request) -> None:
    # Behind Render's proxy the client address is the first X-Forwarded-For entry.
    forwarded = request.headers.get("x-forwarded-for", "")
    client = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now, window = time.monotonic(), _requests[client]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(429, f"Too many questions; the limit is {RATE_LIMIT_PER_MINUTE} per minute.")
    window.append(now)


def source_payload(chunks) -> list[dict]:
    payload = []
    for n, item in enumerate(chunks, start=1):
        section = item.chunk.metadata.get("section", "")
        text = item.chunk.text
        if section and text.startswith(section):  # the label already shows the section path
            text = text[len(section):].lstrip()
        similarity = round(item.similarity, 3) if item.similarity is not None else None
        payload.append({"n": n, "label": citation_label(item), "section": section,
                        "page": item.chunk.metadata.get("page"), "similarity": similarity, "text": text})
    return payload


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(pipeline.retriever.chunks), "llm": pipeline.llm is not None}


@app.post("/api/ask")
def ask(body: AskRequest, request: Request) -> StreamingResponse:
    check_rate_limit(request)
    history = [m.model_dump() for m in body.history[-MAX_HISTORY_MESSAGES:]]

    def events():
        sent_sources, answer = False, ""
        try:
            for answer_so_far, chunks in pipeline.stream_answer(body.question.strip(), history, top_k=body.top_k):
                if not sent_sources:
                    yield json.dumps({"type": "sources", "sources": source_payload(chunks)}) + "\n"
                    sent_sources = True
                delta, answer = answer_so_far[len(answer):], answer_so_far
                if delta:
                    yield json.dumps({"type": "delta", "text": delta}) + "\n"
            yield json.dumps({"type": "done", "answer": answer, "cited": sorted(cited_indices(answer))}) + "\n"
        except Exception:
            logging.exception("Answering failed")
            yield json.dumps({"type": "error", "message": "The server hit an error while answering. Please try again."}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
