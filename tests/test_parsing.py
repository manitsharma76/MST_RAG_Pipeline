from rag.indexing import load_notes, note_files
from rag.loaders import load_file
from rag.parsing.markdown_cleanup import build_notes_file, clean_page
from rag.retriever import tokenize
from rag.search_text import to_search_text

LLAMAPARSE_PAGE = """## **Runtime**

<table><tr><th>Priority Queue</th><th><em>T</em><sub>Extract-Min</sub></th><th>Total</th></tr>
<tr><td>Array</td><td><em>O</em>(<em>V</em>)</td><td><em>O</em>(<em>V</em><sup>2</sup>)</td></tr>
<tr><td>Fibonacci heap<br />[CLRS ch. 19]</td><td><em>O</em>(lg <em>V</em>) (amortized)</td><td>a &amp; b</td></tr></table>

```text
 7      **if** Find-Set(u) ≠ Find-Set(v):
```"""


def test_clean_page_converts_tables_and_strips_bold():
    cleaned = clean_page(LLAMAPARSE_PAGE)
    assert cleaned.splitlines()[0] == "## Runtime"
    assert "| Priority Queue | T_{Extract-Min} | Total |" in cleaned
    assert "|---|---|---|" in cleaned
    assert "| Array | O(V) | O(V^2) |" in cleaned
    assert "| Fibonacci heap [CLRS ch. 19] | O(lg V) (amortized) | a & b |" in cleaned
    assert " 7      if Find-Set(u) ≠ Find-Set(v):" in cleaned  # bold gone, indentation kept
    assert "<" not in cleaned


def test_reviewed_notes_file_round_trips_through_loader(tmp_path):
    text = build_notes_file("lec12.pdf", [(1, "# Title\n\nIntro."), (2, ""), (3, "```\n12\n```")], "LlamaParse")
    path = tmp_path / "lec12.md"
    path.write_text(text)
    docs = load_file(path)
    assert [(d.metadata["source"], d.metadata["page"]) for d in docs] == [("lec12.pdf", 1), ("lec12.pdf", 3)]
    assert "Parsed by" not in docs[0].text  # provenance comment dropped
    assert docs[1].text == "```\n12\n```"  # a lone number inside code is not a page number


def test_reviewed_markdown_replaces_pdf_with_same_stem(tmp_path):
    raw, parsed = tmp_path / "raw", tmp_path / "parsed"
    raw.mkdir()
    parsed.mkdir()
    (raw / "lec12.pdf").write_bytes(b"%PDF-1.4")
    (raw / "extra.md").write_text("# Extra\n\nNotes.")
    (parsed / "lec12.md").write_text(build_notes_file("lec12.pdf", [(1, "# Lec\n\nText.")], "LlamaParse"))

    assert note_files(raw, parsed) == [raw / "extra.md", parsed / "lec12.md"]
    assert [source for source, _ in load_notes(raw, parsed)] == ["extra.md", "lec12.pdf"]


def test_search_text_spells_out_math():
    assert to_search_text("if Find-Set(u) ≠ Find-Set(v):") == "if Find-Set(u) not equal Find-Set(v):"
    assert "not in" in to_search_text("v ∉ S and $v \\notin S$ and v /∈ S")
    assert tokenize(to_search_text(r"$\sum_{e \in T} w(e)$")) == ["sum", "e", "in", "t", "w", "e"]
    assert tokenize(to_search_text("O(E lg V) with V \\ {s}")) == ["o", "e", "lg", "log", "v", "with", "v", "minus", "s"]
    assert "infinity" in to_search_text("v.key = ∞") and "empty set" in to_search_text("T = ∅")
    assert "less than" in to_search_text("w(u, v) < v.key")
