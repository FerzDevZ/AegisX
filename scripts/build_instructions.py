"""Build a lightweight instruction dataset from the User:/AegisX: chat corpus.

Reads every `data/raw/*.txt` file whose paragraphs look like

    User: <question>
    AegisX: <answer>

and writes them to `data/finetune/instructions.jsonl` as Alpaca-style rows:

    {"instruction": "<question>", "output": "<answer>"}

This is the "light and fast" format for SFT/QLoRA fine-tuning (see
notebooks/aegisx_finetune_colab.ipynb): small file, high signal per token,
quick to train on a free T4.

Usage:
    python scripts/build_instructions.py                  # -> data/finetune/instructions.jsonl
    python scripts/build_instructions.py --max-answer 200 # cap answers for speed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/finetune/instructions.jsonl")


def parse_chat_file(path: Path) -> list[dict]:
    """Extract (instruction, output) pairs from one chat-format file."""
    text = path.read_text(encoding="utf-8")
    rows: list[dict] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para.startswith("User:"):
            continue
        # Find the answer boundary marker.
        marker = para.find("\nAegisX:")
        if marker == -1:
            continue
        instruction = para[len("User:") : marker].strip()
        output = para[marker + len("\nAegisX:") :].strip()
        if instruction and output:
            rows.append({"instruction": instruction, "output": output})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build instruction JSONL from chat corpus")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--out", default=str(OUT_FILE))
    parser.add_argument("--max-answer", type=int, default=0, help="truncate answers to N chars (0 = keep full)")
    parser.add_argument("--force", action="store_true", help="overwrite existing output")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"{out_path} exists - skipping (--force to rebuild)")
        return

    all_rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    sources: dict[str, int] = {}
    for path in sorted(Path(args.raw_dir).glob("*.txt")):
        rows = parse_chat_file(path)
        for row in rows:
            if args.max_answer and len(row["output"]) > args.max_answer:
                row["output"] = row["output"][: args.max_answer].rsplit(" ", 1)[0] + "…"
            key = (row["instruction"], row["output"])
            if key in seen:
                continue
            seen.add(key)
            all_rows.append(row)
            sources[path.name] = sources.get(path.name, 0) + 1

    if not all_rows:
        print(f"No User:/AegisX: pairs found under {args.raw_dir} - nothing to write.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_tokens = sum(len(r["instruction"].split()) + len(r["output"].split()) for r in all_rows)
    print(f"✅ Wrote {len(all_rows)} rows -> {out_path}")
    print(f"   size: {out_path.stat().st_size/1024:.1f} KB | ~{total_tokens:,} words")
    print("   by source:")
    for name, count in sorted(sources.items()):
        print(f"     {name}: {count}")
    print("   sample:")
    print(json.dumps(all_rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
