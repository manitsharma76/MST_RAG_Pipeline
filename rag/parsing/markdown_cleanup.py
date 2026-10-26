"""Turn parser Markdown into the plain Markdown the loader and chunker expect.

- `**bold**` markers are removed (they also appear inside code blocks and headings).
- HTML tables become Markdown tables; <sup>/<sub> become ^ and _ (V<sup>2</sup> -> V^2).
"""
import html
import re
from datetime import date

_TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)
_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


def clean_page(markdown: str) -> str:
    text = _TABLE.sub(lambda m: html_table_to_markdown(m.group(0)), markdown)
    text = text.replace("**", "")
    lines = [line.rstrip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_table_to_markdown(table: str) -> str:
    rows = [[_cell_text(cell) for cell in _CELL.findall(row)] for row in _ROW.findall(table)]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n\n" + "\n".join(lines) + "\n\n"


def _cell_text(cell: str) -> str:
    cell = re.sub(r"<br\s*/?>", " ", cell, flags=re.I)
    cell = re.sub(r"<sup>(.*?)</sup>", lambda m: "^" + _group(m.group(1)), cell, flags=re.S | re.I)
    cell = re.sub(r"<sub>(.*?)</sub>", lambda m: "_" + _group(m.group(1)), cell, flags=re.S | re.I)
    cell = html.unescape(_TAG.sub("", cell))
    return " ".join(cell.split()).replace("|", r"\|")


def _group(script: str) -> str:
    script = _TAG.sub("", script).strip()
    return script if len(script) == 1 else "{" + script + "}"


def build_notes_file(source_name: str, pages: list[tuple[int, str]], parser: str) -> str:
    """Assemble the reviewed-notes file: source and provenance comments, then page-marked Markdown."""
    header = [
        f"<!-- source: {source_name} -->",
        f"<!-- Parsed by {parser} on {date.today().isoformat()}. Review this file against the PDF "
        "(pseudocode, tables and $math$); your edits are kept and indexed. -->",
    ]
    body = [f"<!-- page {number} -->\n\n{text}" for number, text in pages if text.strip()]
    return "\n".join(header) + "\n\n" + "\n\n".join(body) + "\n"
