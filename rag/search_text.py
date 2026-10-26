"""Search-only view of chunk text: math symbols and LaTeX spelled out as words.

The embedding model and BM25 see `to_search_text(text)`; the LLM and the UI still get the
exact text. So "Find-Set(u) ≠ Find-Set(v)" is searchable as "not equal", "$\\sum_{e \\in T}$"
as "sum ... in", and "lg" also matches "log".
"""
import re

_REPLACEMENTS = [
    (r"\\notin(?![A-Za-z])|∉|/\s*∈|∈\s*/", " not in "),
    (r"\\neq?(?![A-Za-z])|≠|!=", " not equal "),
    (r"\\leq?(?![A-Za-z])|≤|<=", " less than or equal "),
    (r"\\geq?(?![A-Za-z])|≥|>=", " greater than or equal "),
    (r"\\implies(?![A-Za-z])|=⇒|⇒", " implies "),
    (r"\\in(?![A-Za-z])|∈", " in "),
    (r"\\emptyset(?![A-Za-z])|\\varnothing(?![A-Za-z])|∅", " empty set "),
    (r"\\infty(?![A-Za-z])|∞", " infinity "),
    (r"\\sum(?![A-Za-z])|Σ|∑", " sum "),
    (r"\\alpha(?![A-Za-z])|α", " alpha "),
    (r"\\cup(?![A-Za-z])|∪", " union "),
    (r"\\cap(?![A-Za-z])|∩", " intersection "),
    (r"\\subseteq(?![A-Za-z])|\\subset(?![A-Za-z])|⊆|⊂", " subset "),
    (r"\\setminus(?![A-Za-z])|(?<=\s)\\(?=\s)", " minus "),
    (r"\\to(?![A-Za-z])|\\rightarrow(?![A-Za-z])|→", " to "),
    (r"\\cdot(?![A-Za-z])|·", " times "),
    (r"\\lg(?![A-Za-z])|\blg\b", " lg log "),
    (r"\\(log|min|max)(?![A-Za-z])", r" \1 "),
    (r"<", " less than "),
    (r">", " greater than "),
    (r"\\[A-Za-z]+", " "),  # any other LaTeX command
    (r"[$*_^{}`#|\\]", " "),  # Markdown and LaTeX markup
]
_COMPILED = [(re.compile(pattern), replacement) for pattern, replacement in _REPLACEMENTS]


def to_search_text(text: str) -> str:
    for pattern, replacement in _COMPILED:
        text = pattern.sub(replacement, text)
    return re.sub(r"[ \t]+", " ", text).strip()
