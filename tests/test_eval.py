"""Tests for the AegisX eval harness."""

from aegisx import eval as ev


def test_question_set_balanced():
    langs = [q["lang"] for q in ev.EVAL_QUESTIONS]
    assert langs.count("en") >= 8
    assert langs.count("id") >= 8
    # every question has a prompt and keywords
    for q in ev.EVAL_QUESTIONS:
        assert q["q"].strip()
        assert q["keywords"]
        assert all(kw.islower() for kw in q["keywords"])


def test_coverage_full_hit():
    answer = "SQL injection happens when a query hits the database unsafely."
    assert ev.coverage(answer, ["sql", "query", "database"]) == 1.0


def test_coverage_partial_and_none():
    answer = "port scanning with nmap"
    assert ev.coverage(answer, ["nmap", "port", "database"]) == 2 / 3
    assert ev.coverage("hello world", ["nmap", "port"]) == 0.0


def test_coverage_case_insensitive():
    assert ev.coverage("NMAP and Port scans", ["nmap", "port"]) == 1.0


def test_coverage_empty_keywords():
    assert ev.coverage("anything", []) == 0.0
