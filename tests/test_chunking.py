import pymupdf

from rag.chunking import chunk_documents
from rag.loaders import Document, load_file
from rag.preprocessing import clean_text, find_repeated_lines

PSEUDOCODE = """KRUSKAL(G):
  A = {}
  for each vertex v: MAKE-SET(v)
  sort edges by weight
  for each edge (u, v):
    if FIND-SET(u) != FIND-SET(v):
      A = A + {(u, v)}; UNION(u, v)
  return A"""


def make_docs():
    page1 = "# Minimum Spanning Trees\n\nAn MST connects all vertices with minimum total weight.\n\n## Kruskal's Algorithm\n\nKruskal sorts the edges."
    page2 = f"Kruskal uses union-find to detect cycles.\n\n{PSEUDOCODE}\n\n## Prim's Algorithm\n\nPrim grows a single tree from a root."
    return [
        Document(page1, {"source": "notes.pdf", "page": 1}),
        Document(page2, {"source": "notes.pdf", "page": 2}),
    ]


def test_sections_follow_headings_across_pages():
    chunks = chunk_documents(make_docs(), chunk_size=2000, chunk_overlap=200)
    sections = [c.metadata["section"] for c in chunks]
    assert sections == [
        "Minimum Spanning Trees",
        "Minimum Spanning Trees > Kruskal's Algorithm",
        "Minimum Spanning Trees > Prim's Algorithm",
    ]
    kruskal = chunks[1]
    assert (kruskal.metadata["page"], kruskal.metadata["page_end"]) == (1, 2)
    assert PSEUDOCODE in kruskal.text  # pseudocode block kept whole
    assert kruskal.text.startswith("Minimum Spanning Trees > Kruskal's Algorithm")


def test_algorithm_detection_uses_word_boundaries():
    chunks = chunk_documents(make_docs(), chunk_size=2000, chunk_overlap=200)
    assert chunks[1].metadata["algorithms"] == "Kruskal, Union-Find"
    assert chunks[2].metadata["algorithms"] == "Prim"
    primary = chunk_documents([Document("The primary goal.", {"source": "x.md", "page": None})], 2000, 200)
    assert primary[0].metadata["algorithms"] == ""


def test_long_sections_are_split_with_overlap_and_stable_ids():
    paragraphs = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    docs = [Document(f"## Long\n\n{paragraphs}", {"source": "long.md", "page": None})]
    chunks = chunk_documents(docs, chunk_size=600, chunk_overlap=250)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 600 + len("Long\n\n")
    first_para_of_second = chunks[1].text.split("\n\n")[1]
    assert first_para_of_second in chunks[0].text  # overlap carried over
    assert len({c.id for c in chunks}) == len(chunks)
    assert [c.id for c in chunk_documents(docs, 600, 250)] == [c.id for c in chunks]


def test_clean_text_dehyphenates_and_drops_page_numbers():
    assert clean_text("spann-\ning tree\n\n12\n\n\n\nnext") == "spanning tree\n\nnext"


def test_clean_text_handles_soft_hyphens_and_control_chars():
    assert clean_text("\x02op\u00ad\ntimal gradu\u00adally") == "optimal gradually"


def test_find_repeated_lines_spots_headers():
    pages = [["DSA Notes", f"content about topic {'abcd'[i]}", f"Page {i}"] for i in range(4)]
    assert find_repeated_lines(pages) == {"dsa notes", "page 0"}


def test_pdf_loader_marks_headings_and_strips_running_headers(tmp_path):
    path = tmp_path / "notes.pdf"
    pages = [
        ("Kruskal Algorithm", "Kruskal sorts every edge by weight first.", "It skips edges whose endpoints share a set."),
        ("Prim Algorithm", "Prim grows a tree outward from one vertex.", "A priority queue holds the crossing edges."),
        ("Boruvka Algorithm", "Each component picks its cheapest edge.", "Components merge until one tree is left."),
    ]
    pdf = pymupdf.open()
    for title, line1, line2 in pages:
        page = pdf.new_page()
        page.insert_text((72, 40), "DSA Course Notes", fontsize=9)
        page.insert_text((72, 100), title, fontsize=20)
        page.insert_text((72, 140), line1, fontsize=11)
        page.insert_text((72, 160), line2, fontsize=11)
    pdf.save(path)

    docs = load_file(path)
    assert [d.metadata["page"] for d in docs] == [1, 2, 3]
    assert "# Kruskal Algorithm" in docs[0].text
    assert "Kruskal sorts every edge by weight first." in docs[0].text
    assert "DSA Course Notes" not in docs[0].text


def test_pdf_title_matching_running_header_is_kept_and_joined(tmp_path):
    # MIT notes layout: a small running header repeats the second line of the large page-1 title.
    path = tmp_path / "lecture.pdf"
    pdf = pymupdf.open()
    for i, body in enumerate(["Greedy algorithms make locally best choices.", "Prim grows a tree.", "Kruskal sorts edges."]):
        page = pdf.new_page()
        page.insert_text((72, 30), "Minimum Spanning Tree", fontsize=9)
        if i == 0:
            page.insert_text((72, 100), "Greedy Algorithms and", fontsize=20)
            page.insert_text((72, 124), "Minimum Spanning Tree", fontsize=20)
        page.insert_text((72, 170), body, fontsize=11)
    pdf.save(path)

    docs = load_file(path)
    assert "# Greedy Algorithms and Minimum Spanning Tree" in docs[0].text
    assert all("Minimum Spanning Tree" not in d.text for d in docs[1:])


def test_generic_headings_nest_under_the_preceding_topic():
    text = (
        "# MST\n\n## Prim's Algorithm\n\nPrim grows a tree.\n\n### Pseudocode\n\nwhile Q not empty\n\n"
        "## Correctness\n\nPrim invariant.\n\n## Runtime\n\nO(E lg V).\n\n"
        "## Kruskal's Algorithm\n\nKruskal sorts.\n\n## Runtime\n\nO(E lg E)."
    )
    chunks = chunk_documents([Document(text, {"source": "n.md", "page": None})], 2000, 200)
    assert [c.metadata["section"] for c in chunks] == [
        "MST > Prim's Algorithm",
        "MST > Prim's Algorithm > Pseudocode",
        "MST > Prim's Algorithm > Correctness",
        "MST > Prim's Algorithm > Runtime",
        "MST > Kruskal's Algorithm",
        "MST > Kruskal's Algorithm > Runtime",
    ]


def test_code_block_with_blank_lines_and_hash_comments_stays_whole():
    code = "```text\n1 for v in V\n\n# not a heading\n2   Make-Set(v)\n```"
    docs = [Document(f"## Kruskal\n\nIntro.\n\n{code}\n\nAfter.", {"source": "n.md", "page": None})]
    [chunk] = chunk_documents(docs, 2000, 200)
    assert code in chunk.text
    assert chunk.metadata["section"] == "Kruskal"


def test_oversized_code_block_and_table_are_split_cleanly():
    code = "```text\n" + "\n".join(f"{i:2}  step {i} " + "x" * 40 for i in range(1, 31)) + "\n```"
    table = "| a | b |\n|---|---|\n" + "\n".join(f"| row {i} | " + "y" * 40 + " |" for i in range(30))
    docs = [Document(f"## Big\n\n{code}\n\n{table}", {"source": "n.md", "page": None})]
    chunks = chunk_documents(docs, 600, 0)
    code_chunks = [c for c in chunks if "```" in c.text]
    table_chunks = [c for c in chunks if "| a | b |" in c.text]
    assert len(code_chunks) > 1 and len(table_chunks) > 1
    assert all(c.text.count("```") == 2 for c in code_chunks)  # every piece re-fenced
    assert all("|---|---|" in c.text for c in table_chunks)  # every piece keeps the header
    body = "".join(c.text for c in code_chunks)
    assert all(f"step {i} " in body for i in range(1, 31))  # nothing lost


def test_generic_headings_skip_pseudocode_parent():
    # LlamaParse layout: "### Correctness" right after "### Kruskal's Algorithm Pseudocode".
    text = ("## Kruskal's Algorithm\n\nIntro.\n\n### Kruskal's Algorithm Pseudocode\n\n```\n1 sort\n```\n\n"
            "### Correctness\n\nClaim 3.\n\n## Prim's Algorithm\n\nGrow.\n\n### Prim's Algorithm Psuedocode\n\n```\n1 Q = V\n```\n\n"
            "## Correctness\n\nInvariants.")
    chunks = chunk_documents([Document(text, {"source": "n.md", "page": None})], 2000, 200)
    assert [c.metadata["section"] for c in chunks] == [
        "Kruskal's Algorithm",
        "Kruskal's Algorithm > Kruskal's Algorithm Pseudocode",
        "Kruskal's Algorithm > Correctness",
        "Prim's Algorithm",
        "Prim's Algorithm > Prim's Algorithm Psuedocode",
        "Prim's Algorithm > Correctness",
    ]
