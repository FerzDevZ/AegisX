"""Tests for scripts/expand_instructions.py paraphrase generation."""

from scripts.expand_instructions import en_variants, id_variants, expand_rows


def test_en_what_is_variants():
    vs = en_variants("What is SQL injection?")
    assert "Explain SQL injection." in vs
    assert "Can you explain SQL injection?" in vs
    assert "What does SQL injection mean?" in vs


def test_en_how_do_i_variants():
    vs = en_variants("How do I enumerate subdomains?")
    assert "How can I enumerate subdomains?" in vs
    assert "What is the best way to enumerate subdomains?" in vs
    assert "Explain how to enumerate subdomains." in vs


def test_en_compound_question_smoothed():
    vs = en_variants("What is SQL injection and how do I test for it?")
    # Second clause should be smoothed, no dangling "how do I".
    for v in vs:
        assert "how do I test" not in v
    assert any("how to test for it" in v for v in vs)


def test_en_difference_no_broken_article():
    vs = en_variants("What is the difference between XSS and CSRF?")
    # "Explain The difference" would be wrong capitalization.
    assert all("Explain the difference" not in v or "The difference" not in v for v in vs)
    assert any(v.startswith("Explain the difference") for v in vs)
    assert any(v.startswith("Compare ") for v in vs)


def test_id_apa_variants():
    vs = id_variants("Apa itu SQL injection?")
    assert "Apa yang dimaksud dengan SQL injection?" in vs
    assert "Jelaskan apa itu SQL injection." in vs


def test_id_bagaimana_variants():
    vs = id_variants("Bagaimana cara menguji IDOR?")
    assert "Bagaimana menguji IDOR?" in vs
    assert "Tolong jelaskan cara menguji IDOR." in vs


def test_expand_rows_grows_and_dedupes():
    rows = [
        {"instruction": "What is nmap?", "output": "A port scanner."},
        {"instruction": "What is SQL injection?", "output": "A query injection bug."},
    ]
    expanded = expand_rows(rows, target=20, max_per_row=6)
    assert len(expanded) > len(rows)
    # No exact duplicate instruction/answer pairs.
    pairs = [(r["instruction"], r["output"]) for r in expanded]
    assert len(pairs) == len(set(pairs))
    # Every original is preserved.
    assert rows[0] in expanded
    assert rows[1] in expanded
