"""Publish frontend/ as a static Hugging Face Space (free on every account) that talks to the backend.

    python scripts/deploy_frontend.py --space Mshxrma/Test-Retrieval-Pipeline \
        --api-url https://mst-rag-api.onrender.com --private

Uploads index.html with the --api-url written into it, and a README with the static-Space header.
The URL is inlined rather than loaded from config.js: a private Space signs only the page request,
so a separate config.js request is refused and the page would have no backend.
The backend must allow the Space's origin (ALLOWED_ORIGINS on Render), which this script prints.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi  # noqa: E402

from config import ROOT  # noqa: E402

README = """---
title: MST Notes Assistant
emoji: 🔗
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
---

Question answering over MIT 6.046J Lecture 12 notes on minimum spanning trees. Every answer is
grounded in the notes and cites its passages. This Space is the interface; retrieval and generation
run on a separate backend ({api_url}).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="Space id, e.g. your-username/your-space")
    parser.add_argument("--api-url", required=True, help="backend base URL, e.g. https://mst-rag-api.onrender.com")
    parser.add_argument("--private", action="store_true", help="only you (logged in) can open the Space")
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    if not api_url.startswith("https://"):
        sys.exit("--api-url must be https:// (a Space page can't call an http:// backend)")

    api = HfApi()
    print(f"Logged in to Hugging Face as {api.whoami()['name']}")
    url = api.create_repo(args.space, repo_type="space", space_sdk="static", private=args.private, exist_ok=True)

    page = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    config_tag = '<script src="config.js"></script>'
    if page.count(config_tag) != 1:
        sys.exit(f"frontend/index.html must contain {config_tag} exactly once")
    page = page.replace(config_tag, f"<script>window.MST_API_URL = {json.dumps(api_url)};</script>")

    operations = [
        CommitOperationAdd("index.html", page.encode("utf-8")),
        CommitOperationAdd("README.md", README.format(api_url=api_url).encode()),
    ]
    existing = set(api.list_repo_files(args.space, repo_type="space"))
    operations += [CommitOperationDelete(name) for name in ("config.js", "style.css") if name in existing]
    commit = api.create_commit(args.space, repo_type="space", operations=operations,
                               commit_message=f"Deploy frontend (backend: {api_url})")
    owner, name = args.space.split("/")
    origin = f"https://{owner.lower()}-{name.lower().replace('_', '-').replace('.', '-')}.static.hf.space"
    print(f"Space: {url}\nCommit: {commit.commit_url}")
    print(f"\nThe page is served from {origin}; make sure ALLOWED_ORIGINS on the backend includes it.")


if __name__ == "__main__":
    main()
