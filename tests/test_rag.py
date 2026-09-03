import pytest

from aegisx.rag import CorpusIndex


@pytest.fixture
def index(tmp_path):
    (tmp_path / "owasp.txt").write_text(
        "SQL injection occurs when user input is concatenated into a query.\n\n"
        "The fix is parameterized queries.\n\n"
        "Cross site scripting injects client side scripts.\n\n"
        "Server side request forgery reaches internal services.\n\n"
        "XML external entity reads local files.\n",
        encoding="utf-8",
    )
    (tmp_path / "recon.txt").write_text(
        "Nmap scans ports on a target host.\n\n"
        "Subdomain enumeration finds hidden attack surface.\n\n"
        "Certificate transparency logs reveal subdomains.\n",
        encoding="utf-8",
    )
    idx = CorpusIndex(chunk_chars=200, overlap_chars=20)
    assert idx.add_dir(str(tmp_path)) == 2
    return idx


def test_chunks_have_sources(index):
    assert len(index) >= 3
    assert all(c.source in {"owasp.txt", "recon.txt"} for c in index.chunks)


def test_retrieve_returns_relevant_chunk(index):
    results = index.retrieve("sql injection parameterized query", top_k=2)
    assert results
    assert "SQL injection" in results[0].text
    assert results[0].source == "owasp.txt"


def test_retrieve_other_topic(index):
    results = index.retrieve("nmap port scan host", top_k=2)
    assert results
    assert results[0].source == "recon.txt"


def test_retrieve_empty_query(index):
    assert index.retrieve("") == []
    assert index.retrieve("   ") == []


def test_retrieve_top_k_limit(index):
    assert len(index.retrieve("the", top_k=1)) <= 1


def test_format_context_has_sources(index):
    results = index.retrieve("sql injection", top_k=1)
    ctx = index.format_context(results)
    assert "[source: owasp.txt" in ctx


def test_long_paragraph_hard_split(tmp_path):
    from pathlib import Path

    p = Path(tmp_path) / "long.txt"
    p.write_text(("alpha beta " * 300), encoding="utf-8")
    idx = CorpusIndex(chunk_chars=150, overlap_chars=30)
    idx.add_file(p)
    assert len(idx) > 1
    # Retrieval should find the term in at least one chunk.
    results = idx.retrieve("alpha", top_k=1)
    assert results