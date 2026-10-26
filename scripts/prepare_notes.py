"""Convert each PDF in data/raw/ into reviewed Markdown in data/parsed/ using LlamaParse.

    python scripts/prepare_notes.py                       # PDFs without a data/parsed/ file yet
    python scripts/prepare_notes.py --force               # regenerate (overwrites your review edits)
    python scripts/prepare_notes.py --tier cost_effective # cheaper LlamaParse tier

LlamaParse keeps pseudocode, tables and math (≠, Σ, superscripts) that text extraction
loses, but it reads the page with a model and can occasionally "correct" the notes.
Review the generated file against the PDF, fix anything that differs, and commit it.
scripts/build_index.py then indexes it in place of the PDF (no LlamaParse key needed there).

Needs LLAMA_CLOUD_API_KEY in .env. The PDF is uploaded to LlamaIndex. Responses are cached
in parsed/llamaparse/, so regenerating from the same tier costs no credits.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT, settings  # noqa: E402
from rag.parsing import llamaparse  # noqa: E402
from rag.parsing.markdown_cleanup import build_notes_file, clean_page  # noqa: E402

CACHE_DIR = ROOT / "parsed" / "llamaparse"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default="agentic", choices=llamaparse.TIERS)
    parser.add_argument("--force", action="store_true", help="overwrite existing files in data/parsed/")
    parser.add_argument("--refresh", action="store_true", help="re-run LlamaParse instead of using its cache")
    args = parser.parse_args()

    pdfs = sorted(settings.raw_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {settings.raw_dir}")
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)

    for pdf in pdfs:
        target = settings.parsed_dir / f"{pdf.stem}.md"
        if target.exists() and not args.force:
            print(f"skip   {pdf.name}: {target.relative_to(ROOT)} exists (may hold your review edits; --force to overwrite)")
            continue
        pages = llamaparse.parse_pdf(pdf, args.tier, CACHE_DIR, refresh=args.refresh)
        cleaned = [(p["page_number"], clean_page(p.get("markdown", ""))) for p in pages]
        text = build_notes_file(pdf.name, cleaned, f"LlamaParse (tier: {args.tier})")
        target.write_text(text, encoding="utf-8")

        code_blocks = text.count("```") // 2
        tables = len(re.findall(r"^\|[\s:|-]+\|$", text, re.M))
        math = len(re.findall(r"\$\$.+?\$\$|\$[^$\n]+\$", text))
        print(f"wrote  {target.relative_to(ROOT)}: {len(cleaned)} pages, {code_blocks} code blocks, "
              f"{tables} tables, {math} math expressions")

    print("\nNext: review each file against its PDF (pseudocode, tables, $math$), then run "
          "python scripts/build_index.py --rebuild")


if __name__ == "__main__":
    main()
