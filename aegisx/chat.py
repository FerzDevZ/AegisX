"""Chat with AegisX-Mini.

Usage:
    python -m aegisx.chat --model checkpoints/aegisx-mini/model.pt \
        --tokenizer checkpoints/aegisx-mini/tokenizer.json
    python -m aegisx.chat --model ... --prompt "nmap scan plan" --max-new-tokens 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from aegisx.model import GPT
from aegisx.tokenizer import ByteLevelBPETokenizer

DEFAULT_PROMPT = (
    "You are AegisX, a cybersecurity AI assistant. "
    "Help with recon, scanning, defense, and bug bounty methodology.\n\nUser: "
)


def generate(
    model_path: str,
    tokenizer_path: str,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    device: str = "cpu",
) -> str:
    model = GPT.load(model_path, device=device)
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
    model.eval()

    ids = tokenizer.encode(prompt, add_special_tokens=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
    gen_ids = out[0].tolist()
    generated = gen_ids[len(ids) :]
    return tokenizer.decode(generated).strip()


def interactive(model_path: str, tokenizer_path: str, **gen_kwargs) -> None:
    print("AegisX-Mini — type your question, Ctrl+C or 'exit' to quit.")
    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break
        prompt = DEFAULT_PROMPT + user + "\n\nAegisX: "
        try:
            print("\nAegisX>", generate(model_path, tokenizer_path, prompt, **gen_kwargs))
        except KeyboardInterrupt:
            print("\n(interrupted)")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with AegisX-Mini")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    if args.prompt:
        text = generate(args.model, args.tokenizer, args.prompt, device=device, **kwargs)
        print(text)
    else:
        interactive(args.model, args.tokenizer, device=device, **kwargs)


if __name__ == "__main__":
    sys.exit(main())