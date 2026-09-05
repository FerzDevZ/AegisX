"""Clean and deduplicate the raw corpus in place.

For every .txt under data/raw:
  - drops pure-URL / banner / separator lines
  - normalizes excessive whitespace
  - drops paragraphs shorter than MIN_CHARS (likely fragments)
  - removes near-duplicate paragraphs (same normalized text within a file
    and across files), keeping the first occurrence

Backs each file up to <file>.bak only if --backup is given (default keeps a
.bak so you can inspect before deleting). Report only by default: pass
--apply to actually rewrite files.

Usage:
    python scripts/clean_corpus.py                 # dry-run report
    python scripts/clean_corpus.py --apply         # rewrite files
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RAW_DIR = Path("data/raw")
MIN_CHARS = 40

# Lines that are pure URLs, markdown rules, or separators.
_NOISE_RE = re.compile(
    r"^(?:"
    r"https?://\S+|www\.\S+|"
    r"[-=_*#]{3,}|"
    r"\[?\[?(?:image|img|figure|table)[^]]*\]?\]?|"
    r"<!--.*-->|"
    r"```|"
    r"---|"
    r"\s*$"
    r")",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Canonical form used for dedup: lowercase, alnum only."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def clean_paragraphs(text: str, min_chars: int = MIN_CHARS) -> list[str]:
    paras = []
    for raw in text.split("\n\n"):
        # Drop noise lines inside the paragraph.
        lines = [ln for ln in raw.splitlines() if not _NOISE_RE.match(ln.strip())]
        para = "\n".join(lines).strip()
        para = re.sub(r"[ \t]+", " ", para)
        if len(para) >= min_chars:
            paras.append(para)
    return paras


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and dedup the raw corpus")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--apply", action="store_true", help="actually rewrite files (default: dry-run)")
    parser.add_argument("--min-chars", type=int, default=40)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    files = sorted(raw_dir.glob("*.txt"))
    if not files:
        print(f"No .txt files under {raw_dir}")
        sys.exit(1)

    min_chars = args.min_chars

    seen_global: set[str] = set()
    total_before = total_after = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        total_before += len(text)
        paras = clean_paragraphs(text, min_chars=min_chars)
        kept = []
        kept_norm: set[str] = set()
        for p in paras:
            norm = normalize_text(p)
            if norm in seen_global or norm in kept_norm:
                continue  # duplicate (same file or earlier file)
            kept_norm.add(norm)
            seen_global.add(norm)
            kept.append(p)
        out = "\n\n".join(kept) + "\n"
        total_after += len(out)
        removed = len(paras) - len(kept)
        flag = "DRY-RUN" if not args.apply else "applied"
        print(f"  {path.name}: {len(paras)} -> {len(kept)} paragraphs (-{removed}) [{flag}]")
        if args.apply:
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists():
                bak.write_text(text, encoding="utf-8")
            path.write_text(out, encoding="utf-8")

    print(f"\nTotal size: {total_before / 1024:.0f} KB -> {total_after / 1024:.0f} KB "
          f"({100 * (1 - total_after / max(1, total_before)):.0f}% smaller)")
    if not args.apply:
        print("Dry run - nothing changed. Re-run with --apply to rewrite. "
              "Each file gets a .bak backup.")


if __name__ == "__main__":
    sys.exit(main())
