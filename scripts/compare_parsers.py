"""Parse the PDFs in data/raw/ with several parsers and compare how well pseudocode,
math and tables survive.

    python scripts/compare_parsers.py
    python scripts/compare_parsers.py --llama-tier cost_effective        # cheaper LlamaParse tier
    python scripts/compare_parsers.py --refresh                           # re-run LlamaParse (uses credits)

Parsers:
  pymupdf     the current loader (rag.loaders), as the baseline
  pdfplumber  layout-preserving text (keeps indentation) + tables as Markdown
  llamaparse  LlamaIndex's cloud parser, v2 API (needs LLAMA_CLOUD_API_KEY; uploads the PDF;
              the response is cached in parsed/llamaparse/ so reruns cost nothing)

Full outputs go to parsed/<parser>/<pdf-stem>.md. The checklist targets the MIT 6.046J
lecture 12 MST notes; checks that don't apply to another PDF simply show ✗.
"""
import argparse
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT, settings  # noqa: E402

OUT_DIR = ROOT / "parsed"
PAGE_BREAK = "\n\n<!-- page {n} -->\n\n"


# --------------------------------------------------------------------------- parsers

def parse_pymupdf(path: Path, args) -> str:
    from rag.loaders import load_file

    return "".join(PAGE_BREAK.format(n=d.metadata["page"]) + d.text for d in load_file(path))


def parse_pdfplumber(path: Path, args) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            boxes = [t.bbox for t in tables]

            def outside_tables(obj, boxes=boxes):
                x = (obj["x0"] + obj["x1"]) / 2
                y = (obj["top"] + obj["bottom"]) / 2
                return not any(x0 <= x <= x1 and top <= y <= bottom for x0, top, x1, bottom in boxes)

            text = page.filter(outside_tables).extract_text(layout=True) if boxes else page.extract_text(layout=True)
            text = textwrap.dedent("\n".join(line.rstrip() for line in text.splitlines()))
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            rendered = [_markdown_table(t.extract()) for t in tables]
            pages.append(PAGE_BREAK.format(n=number) + "\n\n".join([text, *rendered]))
    return "".join(pages)


def parse_llamaparse(path: Path, args) -> str:
    from rag.parsing import llamaparse

    pages = llamaparse.parse_pdf(path, args.llama_tier, OUT_DIR / "llamaparse", refresh=args.refresh)
    return "".join(PAGE_BREAK.format(n=p["page_number"]) + p["markdown"] for p in pages)


PARSERS = {"pymupdf": parse_pymupdf, "pdfplumber": parse_pdfplumber, "llamaparse": parse_llamaparse}


def _markdown_table(rows: list[list]) -> str:
    rows = [[" ".join((cell or "").split()) for cell in row] for row in rows if any(row)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


# --------------------------------------------------------------------------- checklist

def squash(text: str) -> str:
    """Drop whitespace, $, backslashes and braces so plain-text and LaTeX output compare equally."""
    return re.sub(r"[\s$\\{}]", "", text)


def line_with(text: str, anchor: str) -> str:
    for line in text.splitlines():
        if anchor.lower() in line.lower():
            return line.strip()
    return "(not found)"


def snippet(text: str, pattern: str, context: int = 25) -> str:
    match = re.search(rf".{{0,{context}}}(?:{pattern}).{{0,{context}}}", text)
    return match.group(0).strip() if match else "(not found)"


def block_between(text: str, start: str, end: str) -> str:
    match = re.search(re.escape(start) + r"(.*?)" + re.escape(end), text, re.S | re.I)
    return match.group(1) if match else ""


def check_prim(text: str) -> tuple[bool, str]:
    block = block_between(text, "Psuedocode", "In the above pseudocode")
    numbers = [int(m) for m in re.findall(r"^[\s>`|*-]*(\d{1,2})\b", block, re.M) if int(m) <= 12]
    found = sorted(set(numbers))
    # Column where the code starts after the line number, e.g. " 5     v.key = ∞" -> 8.
    starts = {m.end() for m in re.finditer(r"^[\s>`|*-]*\d{1,2}[.)]?\s+", block, re.M)}
    nested = len(starts) >= 3
    missing = [n for n in range(1, 13) if n not in found]
    detail = f"line numbers found {len(found)}/12" + (f" (missing {missing})" if missing else "") + f"; nesting {'kept' if nested else 'lost'}"
    return (not missing and nested), detail


def checklist(text: str) -> list[tuple[str, bool, str]]:
    s = squash(text)
    kruskal_line = line_with(text, "Find-Set(u")
    kruskal_ok = bool(re.search(r"Find-Set\(u\)(≠|neq|ne\(|!=)", squash(kruskal_line)))
    prim_ok, prim_detail = check_prim(text)
    sigma = bool(re.search(r"Σ|∑|sum_", s))
    table_line = line_with(text, "Binary heap")
    table_ok = "|" in table_line and bool(re.search(r"O\(E(lg|log)V\)", squash(table_line)))
    not_in = r"∉|\\notin|/\s*∈|∈\s*/|\\not\s*\\in"
    return [
        ("Kruskal: Find-Set(u) ≠ Find-Set(v)", kruskal_ok, kruskal_line),
        ("Prim pseudocode 1-12 + nesting", prim_ok, prim_detail),
        ("MST objective Σ_{e∈T} w(e)", sigma, snippet(text, r"Σ|∑|\\sum") if sigma else line_with(text, "w(e)")),
        ("Not-in symbol v ∉ S", bool(re.search(not_in, text)), snippet(text, not_in, 15)),
        ("Superscript O(V^2)", bool(re.search(r"O\(V(\^2|²)\)", s)), line_with(text, "Array")),
        ("Primes/stars T', T*", bool(re.search(r"T(\^)?(′|'|prime)", s)) and bool(re.search(r"T(\^)?(\*|∗|ast)", s)), line_with(text, "Let T")),
        ("Runtime table as rows/columns", table_ok, table_line),
        ("Title: Greedy Algorithms and MST", "GreedyAlgorithmsandMinimumSpanningTree" in s, line_with(text, "Greedy Algorithms and")),
        ("Running header removed", "Spring2015" not in s, f"'Spring 2015' appears {text.count('Spring 2015')}x"),
    ]


# --------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parsers", default=",".join(PARSERS), help="comma-separated subset of: " + ", ".join(PARSERS))
    parser.add_argument("--llama-tier", default="agentic",
                        help="LlamaParse v2 tier: fast, cost_effective, agentic, agentic_plus")
    parser.add_argument("--refresh", action="store_true", help="ignore the LlamaParse cache")
    args = parser.parse_args()

    pdfs = sorted(settings.raw_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {settings.raw_dir}")

    for pdf in pdfs:
        print(f"\n=== {pdf.name}")
        results = {}
        for name in args.parsers.split(","):
            try:
                text = PARSERS[name](pdf, args)
            except Exception as exc:
                print(f"  {name}: skipped — {exc}")
                continue
            out = OUT_DIR / name / f"{pdf.stem}.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)
            results[name] = checklist(text)
            print(f"  {name}: {len(text):,} chars -> {out.relative_to(ROOT)}")

        for name, checks in results.items():
            passed = sum(ok for _, ok, _ in checks)
            print(f"\n--- {name}: {passed}/{len(checks)} checks passed")
            for label, ok, evidence in checks:
                print(f"  {'✓' if ok else '✗'} {label:36} {evidence[:110]}")


if __name__ == "__main__":
    main()
