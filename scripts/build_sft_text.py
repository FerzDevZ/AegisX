"""Convert the instruction dataset to chat-format text for AegisX's OWN model.

AegisX-Mini (the from-scratch GPT) has no special chat tokens - it learns
format from raw text. So SFT stage 2 feeds it the same shape it already
knows: "User: ...\nAegisX: ..." paragraphs. This script turns
data/finetune/instructions.jsonl into a plain .txt file that
`python -m aegisx.train --init-from <model.pt>` can train on directly.

Usage:
    python scripts/build_sft_text.py                          # data/finetune/sft_chat.txt
    python scripts/build_sft_text.py --out /tmp/sft.txt --seed 1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_IN = Path("data/finetune/instructions.jsonl")
DEFAULT_OUT = Path("data/finetune/sft_chat.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT chat text from instruction JSONL")
    parser.add_argument("--in", dest="in_file", default=str(DEFAULT_IN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1, help="repeat rows for more training tokens")
    args = parser.parse_args()

    in_path = Path(args.in_file)
    rows = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]
    if not rows:
        raise SystemExit(f"No rows found in {in_path}")

    rng = random.Random(args.seed)
    blocks = []
    for _ in range(args.repeat):
        for r in rows:
            instruction = r.get("instruction", "").strip()
            output = r.get("output", "").strip()
            if not instruction or not output:
                continue
            blocks.append(f"User: {instruction}\nAegisX: {output}\n")

    rng.shuffle(blocks)  # avoid grouping by source file during training
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(blocks), encoding="utf-8")

    words = sum(len(b.split()) for b in blocks)
    print(f"✅ Wrote {len(blocks)} Q&A blocks -> {out_path} ({out_path.stat().st_size/1024:.0f} KB, ~{words:,} words)")
    print("   train with: python -m aegisx.train --data <this dir> --init-from <your model.pt>")


if __name__ == "__main__":
    main()
