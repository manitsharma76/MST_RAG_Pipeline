"""LlamaParse v2 REST client. New accounts can't use the v1 API that llama-cloud-services wraps.

The raw JSON response is cached on disk, so re-running costs no credits.
"""
import json
import os
import time
from pathlib import Path

import httpx

API_URL = "https://api.cloud.llamaindex.ai/api/v2/parse"
TIERS = ("fast", "cost_effective", "agentic", "agentic_plus")


def parse_pdf(path: Path, tier: str, cache_dir: Path, refresh: bool = False) -> list[dict]:
    """Return LlamaParse's pages: [{"page_number": int, "markdown": str, ...}, ...]."""
    cache = cache_dir / f"{path.stem}.{tier}.json"
    if cache.exists() and not refresh:
        result = json.loads(cache.read_text())
    else:
        result = _request(path, tier)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result))
    return result["markdown"]["pages"]


def _request(path: Path, tier: str) -> dict:
    key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not key:
        raise RuntimeError("LLAMA_CLOUD_API_KEY is not set (get a free key at cloud.llamaindex.ai, add it to .env)")
    headers = {"Authorization": f"Bearer {key}"}
    with path.open("rb") as f:
        response = httpx.post(
            f"{API_URL}/upload",
            headers=headers,
            files={"file": (path.name, f, "application/pdf")},
            data={"configuration": json.dumps({"tier": tier, "version": "latest"})},
            timeout=120,
        )
    response.raise_for_status()
    job_id = response.json()["id"]

    deadline = time.monotonic() + 600
    while True:
        job = httpx.get(f"{API_URL}/{job_id}", headers=headers, timeout=60).json()["job"]
        if job["status"] == "COMPLETED":
            break
        if job["status"] not in {"PENDING", "RUNNING", "QUEUED"}:
            raise RuntimeError(f"LlamaParse job {job_id} ended as {job['status']}: {job.get('error_message')}")
        if time.monotonic() > deadline:
            raise RuntimeError(f"LlamaParse job {job_id} still {job['status']} after 10 minutes")
        time.sleep(5)

    result = httpx.get(f"{API_URL}/{job_id}", headers=headers, params={"expand": "markdown"}, timeout=120).json()
    pages = (result.get("markdown") or {}).get("pages") or []
    if not pages or not any(p.get("markdown", "").strip() for p in pages):
        raise RuntimeError(f"LlamaParse job {job_id} returned no Markdown")
    return result
