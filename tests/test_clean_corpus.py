"""Tests for scripts/clean_corpus.py cleaning + dedup logic."""

from scripts.clean_corpus import clean_paragraphs, normalize_text


def test_drops_url_only_lines():
    text = "https://example.com/foo\n\nReal paragraph about SQL injection that is long enough to keep here.\n"
    paras = clean_paragraphs(text)
    assert len(paras) == 1
    assert "Real paragraph" in paras[0]


def test_drops_markdown_rules_and_short_fragments():
    text = "---\n```\n\nshort\n\nThis paragraph is comfortably long enough to survive the minimum length filter.\n"
    paras = clean_paragraphs(text)
    assert len(paras) == 1


def test_normalize_text_lowercases_and_strips():
    assert normalize_text("SQL Injection!") == "sql injection"


def test_clean_paragraphs_removes_inline_noise():
    text = "First line of a real paragraph with substance.\n<!-- comment -->\nhttps://x.dev\nSecond line that keeps it above min length.\n"
    paras = clean_paragraphs(text)
    assert len(paras) == 1
    assert "comment" not in paras[0]
    assert "https" not in paras[0]
