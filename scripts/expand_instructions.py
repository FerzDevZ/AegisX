"""Expand the instruction dataset with natural paraphrased questions.

Reads data/finetune/instructions.jsonl (or --in) and writes a larger version
(--out, same file by default). For every row it generates a few language-aware
question variants that ask the same thing in different words, keeping the
original answer. Filters out anything unnatural or duplicated, so the model
learns that many phrasings map to one correct answer.

Only deterministic, template-based transformations are used - no LLM involved,
so results are reproducible and cheap.

Usage:
    python scripts/expand_instructions.py                    # expand in place
    python scripts/expand_instructions.py --target 1200      # aim for >= N rows
    python scripts/expand_instructions.py --out /tmp/expanded.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_IN = Path("data/finetune/instructions.jsonl")


def _cap_first(text: str) -> str:
    # Don't capitalize leading articles ("Explain the difference ...").
    if re.match(r"^(the|a|an)\s", text, re.I):
        return text
    return text[:1].upper() + text[1:] if text else text


def _strip_q(text: str) -> str:
    return text.strip().rstrip("?").strip()


# --------------------------------------------------------------------------- #
# Paraphrase rules. Each rule receives the full instruction and yields zero or
# more candidate variants. Rules are written so the transformed question stays
# grammatical and on-topic; invalid/duplicate candidates are filtered later.
# --------------------------------------------------------------------------- #
def en_variants(q: str) -> list[str]:
    qq = q.strip()
    out: list[str] = []

    # "What is X?" / "What are X?"
    m = re.match(r"^[Ww]hat (?:is|are) (.+?)\??$", qq)
    if m:
        rest = _strip_q(m.group(1))
        # guard: don't mangle comparative/multi-part questions
        simple = not re.search(r"\b(between|difference|role of|importance of)\b", rest, re.I)
        # Compound question like "What is X and how do I test it?": smooth the
        # second clause so variants read naturally ("how do I" -> "how to").
        has_second_clause = re.search(r"\b(and|but) how (do|can|would) (I|you)\b", rest, re.I)
        rest_smooth = re.sub(r"\b(and|but) how (do|can|would) (I|you)\b", r"and how to", rest, flags=re.I)
        # "What is X?" -> "Explain X."
        out.append(f"Explain {_cap_first(rest_smooth)}.")
        # "What is X?" -> "Can you explain X?"
        out.append(f"Can you explain {rest_smooth}?")
        if simple and not has_second_clause:
            # "What is X?" -> "What does X mean?"
            out.append(f"What does {rest} mean?")
        if not has_second_clause and not rest.lower().startswith(("the difference", "a difference")):
            # "What is X?" -> "Tell me about X."
            out.append(f"Tell me about {rest}.")

    # "How do I X?" / "How do you X?"
    m = re.match(r"^[Hh]ow (?:do|can) (?:I|you) (.+?)\??$", qq)
    if m:
        rest = _strip_q(m.group(1))
        # -> "How can I X?" (only when original was "How do I")
        if qq.startswith(("How do", "how do")):
            out.append(f"How can I {rest}?")
        # -> "What is the best way to X?"
        out.append(f"What is the best way to {rest}?")
        # -> "Explain how to X."
        out.append(f"Explain how to {rest}.")
        # -> "Can you show me how to X?"
        out.append(f"Can you show me how to {rest}?")

    # "What is the difference between A and B?"
    m = re.match(r"^[Ww]hat is the difference between (.+?) and (.+?)\??$", qq)
    if m:
        a, b = _strip_q(m.group(1)), _strip_q(m.group(2))
        out.append(f"Compare {a} and {b}.")
        out.append(f"How are {a} and {b} different?")

    return out


def id_variants(q: str) -> list[str]:
    qq = q.strip()
    out: list[str] = []

    # "Apa itu X?"
    m = re.match(r"^[Aa]pa itu (.+?)\??$", qq)
    if m:
        rest = _strip_q(m.group(1))
        out.append(f"Apa yang dimaksud dengan {rest}?")
        out.append(f"Jelaskan apa itu {rest}.")
        # avoid double question on long compound questions
        if not re.search(r"\b(dan|serta)\b", rest, re.I) or len(rest) < 60:
            out.append(f"Tolong jelaskan tentang {rest}.")
    # "Apa perbedaan antara A dan B?" / "Apa perbedaan A dan B?"
    m = re.match(r"^[Aa]pa perbedaan (?:antara )?(.+?) dan (.+?)\??$", qq)
    if m:
        a, b = _strip_q(m.group(1)), _strip_q(m.group(2))
        out.append(f"Jelaskan perbedaan antara {a} dan {b}.")
        out.append(f"Bandingkan {a} dengan {b}.")

    # "Bagaimana cara X?" -> drop "cara" or wrap politely
    m = re.match(r"^[Bb]agaimana cara (.+?)\??$", qq)
    if m:
        rest = _strip_q(m.group(1))
        out.append(f"Bagaimana {rest}?")
        out.append(f"Tolong jelaskan cara {rest}.")
        out.append(f"Apa langkah-langkah untuk {rest}?")
    # "Bagaimana X?" (no "cara")
    m = re.match(r"^[Bb]agaimana (.+?)\??$", qq)
    if m and not qq.lower().startswith("bagaimana cara"):
        rest = _strip_q(m.group(1))
        out.append(f"Jelaskan bagaimana {rest}.")

    # "Jelaskan X."
    m = re.match(r"^[Jj]elaskan (.+?)[\.\?]?$", qq)
    if m:
        rest = _strip_q(m.group(1))
        out.append(f"Tolong jelaskan {rest}.")
        out.append(f"Bisakah kamu menjelaskan {rest}?")

    return out


def variants_for(q: str) -> list[str]:
    """Pick the rule set by language sniffing on the instruction."""
    id_words = {
        "apa", "itu", "bagaimana", "yang", "untuk", "dengan", "dan", "di",
        "dari", "adalah", "pada", "tolong", "jelaskan", "perbedaan", "antara",
    }
    words = set(q.lower().split())
    return id_variants(q) if (words & id_words) else en_variants(q)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def expand_rows(rows: list[dict], target: int = 0, max_per_row: int = 6) -> list[dict]:
    """Return rows + paraphrased variants, deduplicated, up to target size."""
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(row: dict) -> bool:
        key = (row["instruction"], row["output"])
        if key in seen:
            return False
        seen.add(key)
        out.append(row)
        return True

    # Seed with the original rows.
    for row in rows:
        add(row)

    # Then add paraphrases, round-robin so no single topic dominates.
    more = True
    while more and (not target or len(out) < target):
        more = False
        for row in rows:
            if target and len(out) >= target:
                break
            originals = {_normalize(x["instruction"]) for x in out}
            added_this_row = 0
            for cand in variants_for(row["instruction"]):
                if added_this_row >= max_per_row:
                    break
                if _normalize(cand) in originals:
                    continue
                if len(cand) < 8:
                    continue
                add({"instruction": cand, "output": row["output"]})
                originals.add(_normalize(cand))
                added_this_row += 1
                more = True
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Paraphrase-expand the instruction dataset")
    parser.add_argument("--in", dest="in_file", default=str(DEFAULT_IN))
    parser.add_argument("--out", default=str(DEFAULT_IN))
    parser.add_argument("--target", type=int, default=0, help="aim for at least N rows (0 = all variants)")
    parser.add_argument("--max-per-row", type=int, default=6)
    args = parser.parse_args()

    in_path = Path(args.in_file)
    rows = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]
    print(f"Loaded {len(rows)} rows from {in_path}")

    expanded = expand_rows(rows, target=args.target, max_per_row=args.max_per_row)

    out_path = Path(args.out)
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in expanded),
        encoding="utf-8",
    )
    print(f"✅ Wrote {len(expanded)} rows -> {out_path} ({out_path.stat().st_size/1024:.0f} KB)")
    print(f"   growth: {len(rows)} -> {len(expanded)} (x{len(expanded)/max(1, len(rows)):.1f})")

    # Sanity report: language split via stopword sniffing.
    id_words = {"apa", "itu", "bagaimana", "yang", "untuk", "dengan", "dan", "di", "dari", "adalah", "pada"}
    n_id = sum(1 for r in expanded if set(r["instruction"].lower().split()) & id_words)
    print(f"   approx ID rows: {n_id} | EN rows: {len(expanded) - n_id}")

    # A few samples.
    print("   sample variants:")
    shown = 0
    for r in expanded:
        if shown >= 5:
            break
        if r["instruction"] not in {x["instruction"] for x in rows}:
            print(f"     - {r['instruction']}")
            shown += 1


if __name__ == "__main__":
    main()
