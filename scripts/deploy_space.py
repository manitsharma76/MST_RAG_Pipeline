"""Deploy the app to a Hugging Face Space.

    python scripts/deploy_space.py --space <hf-username>/mst-rag --private
    python scripts/deploy_space.py --space <hf-username>/mst-rag          # public

Steps: check the LLM key works, create the Space (if needed), store the LLM keys as
encrypted Space secrets, and upload the code, the reviewed notes and a README carrying the
Gradio Space header (the repository README has no header, so it reads well on GitHub).

Note: Hugging Face now requires a paid plan for Gradio Spaces on CPU Basic; free accounts
older than 30 days can use ZeroGPU hardware instead. The default deployment is the static
Space + Render backend (scripts/deploy_frontend.py, render.yaml). Needs a Hugging Face
login with write access (`hf auth login`, or HF_TOKEN in the environment).

Not uploaded: .env and other secrets, .venv, the local index (the Space builds its own
from data/parsed/ at startup), parser caches, and Python/test caches.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import HfApi  # noqa: E402

from config import ROOT, settings  # noqa: E402

IGNORE = [
    ".env", ".env.*", ".venv/**", ".git/**", "index/**", "parsed/**",
    "**/__pycache__/**", "**/*.pyc", ".pytest_cache/**", ".DS_Store", "**/.DS_Store", ".gradio/**", "flagged/**", ".cache/**",
    "README.md", "CLAUDE.md", ".playwright-mcp/**",
]
SPACE_README = """---
title: MST Notes Assistant
emoji: 🔗
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
---

Question answering over MIT 6.046J Lecture 12 notes on minimum spanning trees, grounded in the
notes with page-level citations.
"""
KEY_NAMES = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}


def check_llm_key(provider: str) -> tuple[str, str]:
    name = KEY_NAMES[provider]
    key = os.getenv(name, "").strip()
    if not key:
        sys.exit(f"{name} is not set in .env")
    if provider == "groq":
        if not key.startswith("gsk_"):
            sys.exit(f"{name} doesn't look like a Groq key (they start with 'gsk_'). Check .env.")
        from groq import AuthenticationError, Groq

        try:
            Groq(api_key=key).models.list()
        except AuthenticationError:
            sys.exit(f"Groq rejected {name} (401 Invalid API Key). Create a new key at console.groq.com/keys.")
    return name, key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="Space id, e.g. your-username/mst-rag")
    parser.add_argument("--private", action="store_true", help="only you (logged in) can open the Space")
    args = parser.parse_args()

    if not any(settings.parsed_dir.glob("*.md")) and not any(settings.raw_dir.glob("*.pdf")):
        sys.exit("No notes to deploy: data/parsed/ and data/raw/ are empty.")
    key_name, key = check_llm_key(settings.llm_provider)

    api = HfApi()
    user = api.whoami()["name"]
    print(f"Logged in to Hugging Face as {user}")

    url = api.create_repo(args.space, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
    print(f"Space ready: {url}")

    api.add_space_secret(args.space, key_name, key)
    api.add_space_variable(args.space, "LLM_PROVIDER", settings.llm_provider)
    print(f"Stored {key_name} as an encrypted Space secret")
    for backup in (n.strip() for n in settings.llm_backup_providers.split(",") if n.strip()):
        backup_key = os.getenv(KEY_NAMES.get(backup, ""), "").strip()
        if backup_key and KEY_NAMES[backup] != key_name:
            api.add_space_secret(args.space, KEY_NAMES[backup], backup_key)
            print(f"Stored {KEY_NAMES[backup]} (backup provider) as an encrypted Space secret")

    commit = api.upload_folder(
        folder_path=ROOT, repo_id=args.space, repo_type="space",
        ignore_patterns=IGNORE,
        commit_message="Deploy MST RAG app",
    )
    api.upload_file(path_or_fileobj=SPACE_README.encode(), path_in_repo="README.md", repo_id=args.space,
                    repo_type="space", commit_message="Space README")
    print(f"Uploaded: {commit.commit_url}")
    print(f"\nThe Space is building. Watch the logs at https://huggingface.co/spaces/{args.space}?logs=container")


if __name__ == "__main__":
    main()
