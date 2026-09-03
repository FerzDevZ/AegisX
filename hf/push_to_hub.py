"""Upload a trained AegisX-Mini checkpoint to Hugging Face Hub.

Usage:
    HF_TOKEN=hf_xxx python hf/push_to_hub.py \
        --repo youruser/aegisx-mini --checkpoint checkpoints/aegisx-mini
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Push AegisX-Mini to Hugging Face Hub")
    parser.add_argument("--repo", required=True, help="e.g. youruser/aegisx-mini")
    parser.add_argument("--checkpoint", required=True, help="local checkpoint dir (model.pt + tokenizer.json)")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN env var required. Create one at https://huggingface.co/settings/tokens")
        sys.exit(1)

    ckpt = Path(args.checkpoint)
    for required in ("model.pt", "tokenizer.json"):
        if not (ckpt / required).exists():
            print(f"Missing {required} in {ckpt}")
            sys.exit(1)

    api = HfApi(token=token)
    api.create_repo(args.repo, exist_ok=True, private=args.private)
    api.upload_folder(
        repo_id=args.repo,
        folder_path=str(ckpt),
        repo_type="model",
        commit_message="AegisX-Mini checkpoint",
    )
    print(f"Pushed {ckpt} -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    sys.exit(main())