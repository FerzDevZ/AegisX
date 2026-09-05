"""Build preference pairs (prompt/chosen/rejected) for stage-3 DPO alignment.

Reads the instruction dataset built by build_instructions.py and produces
data/finetune/preferences.jsonl: one in-scope pair per instruction (real
answer chosen, refusal rejected) plus out-of-scope pairs (refusal chosen,
harmful placeholder rejected). Deterministic via --seed.

Usage:
    python scripts/build_preferences.py \
        --instructions data/finetune/instructions.jsonl \
        --out data/finetune/preferences.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Scripts run from repo root: make the aegisx package importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aegisx.preferences import build_pairs, write_pairs_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs")
    parser.add_argument("--instructions", default="data/finetune/instructions.jsonl")
    parser.add_argument("--out", default="data/finetune/preferences.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for line in Path(args.instructions).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        print(f"No rows in {args.instructions} — run build_instructions.py first.")
        return 1

    pairs = build_pairs(rows, seed=args.seed)
    n = write_pairs_jsonl(pairs, args.out)
    n_refusal = sum(1 for p in pairs if any(k in p["chosen"] for k in ("tidak bisa", "tidak akan", "Maaf", "di luar cakupan")))
    print(f"Wrote {n} pairs -> {args.out} (out-of-scope refusal pairs: {n_refusal})")
    return 0


if __name__ == "__main__":
    sys.exit(main())