"""Evaluate AegisX-Mini against a fixed set of questions (EN + ID).

Runs the trained model over ~20 curated questions and scores each answer by
keyword coverage, so every retrain produces a comparable before/after number
instead of a vibe check.

Usage:
    python -m aegisx.eval --model checkpoints/aegisx-mini/model.pt \
        --tokenizer checkpoints/aegisx-mini/tokenizer.json
    python -m aegisx.eval --model ... --tokenizer ... --max-questions 4
"""

from __future__ import annotations

import argparse
import re
import sys

import torch

from aegisx.model import GPT
from aegisx.tokenizer import ByteLevelBPETokenizer

DEFAULT_SYSTEM = (
    "You are AegisX, a cybersecurity AI assistant. "
    "Help with recon, scanning, defense, and bug bounty methodology."
)

# Fixed question set. Keep keywords as lowercase single tokens that should
# plausibly appear in a correct answer. Broad on purpose: this measures
# progress across retrains, not absolute quality.
EVAL_QUESTIONS: list[dict] = [
    # --- English ---
    {"lang": "en", "q": "What is SQL injection?", "keywords": ["sql", "injection", "query", "database"]},
    {"lang": "en", "q": "How do I enumerate subdomains?", "keywords": ["subdomain", "dns", "brute", "wordlist"]},
    {"lang": "en", "q": "What is XSS and how does it work?", "keywords": ["xss", "script", "cross", "browser"]},
    {"lang": "en", "q": "How do I scan ports with nmap?", "keywords": ["nmap", "port", "scan", "service"]},
    {"lang": "en", "q": "What is CSRF and how do I prevent it?", "keywords": ["csrf", "token", "request", "state"]},
    {"lang": "en", "q": "What is broken access control?", "keywords": ["access", "control", "authorization", "idor", "privilege"]},
    {"lang": "en", "q": "What is in the OWASP Top 10?", "keywords": ["owasp", "injection", "broken", "exposure"]},
    {"lang": "en", "q": "How do I write a pentest report?", "keywords": ["report", "finding", "severity", "remediation", "executive"]},
    {"lang": "en", "q": "How do I defend against phishing?", "keywords": ["phish", "email", "awareness", "training", "link"]},
    {"lang": "en", "q": "What is the 3-2-1 backup rule?", "keywords": ["backup", "copy", "offsite", "recovery"]},
    # --- Bahasa Indonesia ---
    {"lang": "id", "q": "Apa itu SQL injection?", "keywords": ["sql", "injeksi", "injection", "query", "database"]},
    {"lang": "id", "q": "Bagaimana cara enumerasi subdomain?", "keywords": ["subdomain", "dns", "enumerasi", "brute"]},
    {"lang": "id", "q": "Apa itu XSS dan bagaimana cara kerjanya?", "keywords": ["xss", "script", "cross", "browser"]},
    {"lang": "id", "q": "Bagaimana cara scanning port dengan nmap?", "keywords": ["nmap", "port", "scan", "layanan"]},
    {"lang": "id", "q": "Apa itu CSRF dan bagaimana cara mencegahnya?", "keywords": ["csrf", "token", "request"]},
    {"lang": "id", "q": "Apa itu broken access control?", "keywords": ["access", "control", "otorisasi", "idor", "privilege"]},
    {"lang": "id", "q": "Jelaskan OWASP Top 10.", "keywords": ["owasp", "injeksi", "broken", "exposure"]},
    {"lang": "id", "q": "Bagaimana cara menulis laporan pentest?", "keywords": ["laporan", "temuan", "severity", "remediasi", "eksekutif"]},
    {"lang": "id", "q": "Bagaimana cara mencegah phishing?", "keywords": ["phish", "email", "kesadaran", "pelatihan", "tautan"]},
    {"lang": "id", "q": "Apa itu aturan backup 3-2-1?", "keywords": ["backup", "salinan", "offsite", "pemulihan"]},
]

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def coverage(answer: str, keywords: list[str]) -> float:
    """Fraction of keywords present in the answer (case-insensitive)."""
    if not keywords:
        return 0.0
    ans_tokens = _tokenize(answer)
    matched = sum(1 for kw in keywords if kw in ans_tokens)
    return matched / len(keywords)


def generate_once(
    model: GPT,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 90,
    temperature: float = 0.5,
    top_k: int = 50,
) -> str:
    """Generate a reply from an already-loaded model (fast for eval loops)."""
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(out[0].tolist()[len(ids) :]).strip()


def run_eval(
    model_path: str,
    tokenizer_path: str,
    device: str = "cpu",
    max_questions: int = 0,
) -> list[dict]:
    """Evaluate every question (or the first max_questions) and return rows."""
    model = GPT.load(model_path, device=device)
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
    model.eval()

    questions = EVAL_QUESTIONS
    if max_questions and max_questions < len(questions):
        questions = questions[:max_questions]

    rows = []
    for q in questions:
        prompt = f"{DEFAULT_SYSTEM}\n\nUser: {q['q']}\n\nAegisX: "
        try:
            answer = generate_once(model, tokenizer, prompt, device=device)
        except Exception as exc:  # keep the run alive on a single bad question
            answer = f"<error: {exc}>"
        cov = coverage(answer, q["keywords"])
        rows.append(
            {
                "lang": q["lang"],
                "q": q["q"],
                "answer": answer,
                "keywords": q["keywords"],
                "coverage": cov,
            }
        )
    return rows


def print_report(rows: list[dict]) -> dict:
    print(f"{'#':>2} {'lang':<4} {'coverage':>8}  question")
    for i, row in enumerate(rows, 1):
        print(f"{i:>2} {row['lang']:<4} {row['coverage'] * 100:>7.0f}%  {row['q']}")

    def avg(rows_subset: list[dict]) -> float:
        return sum(r["coverage"] for r in rows_subset) / len(rows_subset) if rows_subset else 0.0

    en = [r for r in rows if r["lang"] == "en"]
    id_ = [r for r in rows if r["lang"] == "id"]
    report = {
        "overall": avg(rows),
        "en": avg(en),
        "id": avg(id_),
        "n_questions": len(rows),
    }
    print("\n" + "-" * 46)
    print(f"overall avg coverage : {report['overall'] * 100:5.1f}%  ({report['n_questions']} questions)")
    print(f"EN avg coverage      : {report['en'] * 100:5.1f}%  ({len(en)} questions)")
    print(f"ID avg coverage      : {report['id'] * 100:5.1f}%  ({len(id_)} questions)")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AegisX-Mini on fixed questions")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--max-questions", type=int, default=0, help="0 = run all")
    parser.add_argument("--max-new-tokens", type=int, default=90)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = run_eval(args.model, args.tokenizer, device=device, max_questions=args.max_questions)
    print_report(rows)


if __name__ == "__main__":
    sys.exit(main())
