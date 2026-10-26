import re
import unicodedata
from collections import Counter

_HYPHEN_BREAK = re.compile(r"(\w)[-­]\n(\w)")  # hard or soft hyphen at a line break
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d+(\s*(/|of)\s*\d+)?\s*$", re.IGNORECASE)
_MANY_BLANKS = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalise unicode, join words hyphenated across lines, drop lone page numbers.

    Line structure is otherwise preserved so pseudocode and tables stay intact.
    """
    text = unicodedata.normalize("NFKC", text).replace(" ", " ")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text).replace("­", "")
    lines, in_fence = [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and _PAGE_NUMBER.match(line):
            continue  # a lone number outside code is a page number; inside code it's a line number
        lines.append(line.rstrip())
    return _MANY_BLANKS.sub("\n\n", "\n".join(lines)).strip()


def normalize_line(line: str) -> str:
    """Key used to spot running headers/footers: ignores digits and case.

    Heading marks are kept, so a large-font title line is not mistaken for a small-font
    running header with the same words.
    """
    return re.sub(r"\d+", "0", line.strip().lower())


def find_repeated_lines(page_lines: list[list[str]], min_ratio: float = 0.6) -> set[str]:
    """Normalised lines that appear on most pages (running headers and footers)."""
    if len(page_lines) < 3:
        return set()
    counts = Counter()
    for lines in page_lines:
        counts.update({normalize_line(l) for l in lines if l.strip() and len(l) < 100})
    threshold = min_ratio * len(page_lines)
    return {line for line, n in counts.items() if n >= threshold}
