"""Train AegisX-Mini from scratch on raw text.

CPU-friendly: gradient accumulation, small batch, checkpointing with resume.
Usage:
    python -m aegisx.train --data data/raw --out checkpoints/aegisx-mini
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

from aegisx.model import GPT, ModelConfig
from aegisx.tokenizer import ByteLevelBPETokenizer


def load_texts(data_dir: str | Path) -> list[str]:
    """Load all .txt files under data_dir, split into paragraph-sized chunks."""
    data_dir = Path(data_dir)
    texts: list[str] = []
    for path in sorted(data_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        # Split into non-empty paragraphs to get coherent training units.
        for para in raw.split("\n\n"):
            para = para.strip()
            if len(para) >= 32:  # skip tiny fragments
                texts.append(para)
    if not texts:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")
    return texts


def build_dataset(
    texts: list[str],
    tokenizer: ByteLevelBPETokenizer,
    block_size: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all texts and produce (data, eos) tensors.

    Returns the concatenated token stream; training samples are windows of
    block_size+1 drawn from it.
    """
    rng = random.Random(seed)
    texts = list(texts)  # copy so shuffle does not mutate the caller's list
    rng.shuffle(texts)
    split = max(1, int(len(texts) * 0.95))
    train_texts = texts[:split]
    val_texts = texts[split:]

    eos = tokenizer.special_id("<|endoftext|>")
    train_ids: list[int] = []
    for t in train_texts:
        train_ids.extend(tokenizer.encode(t, add_special_tokens=False))
        train_ids.append(eos)
    val_ids: list[int] = []
    for t in val_texts:
        val_ids.extend(tokenizer.encode(t, add_special_tokens=False))
        val_ids.append(eos)
    return torch.tensor(train_ids, dtype=torch.long), torch.tensor(val_ids, dtype=torch.long)


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random windows (block_size+1) from the token stream.

    Falls back to the single available window when the stream is shorter than
    one full window (tiny datasets or small validation splits), so training
    never crashes on `randint(from >= to)`.
    """
    if len(data) <= block_size + 1:
        if len(data) < 2:
            raise ValueError("Dataset too small: need at least 2 tokens to train.")
        # Single aligned window: x predicts the next token y, same length.
        x = data[:-1].unsqueeze(0)
        y = data[1:].unsqueeze(0)
        return x.to(device), y.to(device)
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
    eval_iters: int = 50,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split_name, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
                    _, loss = model(x, y)
            losses.append(loss.item())
        out[split_name] = float(sum(losses) / len(losses))
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AegisX-Mini from scratch")
    parser.add_argument("--data", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="checkpoints/aegisx-mini")
    parser.add_argument("--tokenizer", type=str, default="", help="load existing tokenizer json (optional)")
    parser.add_argument("--init-from", type=str, default="", help="continue training from this model.pt (SFT/continual pretrain of your own model)")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=100, help="write model_latest.pt every N steps so a dropped session resumes with little loss")
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=10, help="log every N steps")
    parser.add_argument("--early-stop-patience", type=int, default=5, help="stop after N evals without val improvement (0 = disable)")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="min val-loss improvement to count as progress")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA (default: auto-on)")
    parser.add_argument("--history", type=str, default="", help="optional CSV path to log every eval (default: <out>/history.csv)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading corpus from {args.data}")
    texts = load_texts(args.data)
    print(f"      {len(texts)} chunks loaded")

    # --- Optional: continue from a previous checkpoint (SFT / continual) ---
    init_from = Path(args.init_from) if args.init_from else None
    resumed_cfg: ModelConfig | None = None
    if init_from is not None:
        if not init_from.exists():
            raise FileNotFoundError(f"--init-from checkpoint not found: {init_from}")
        ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
        resumed_cfg = ModelConfig.from_dict(ckpt["config"])
        print(f"[init] resuming from {init_from} | {resumed_cfg}")
        print(f"      arch args (--n-layer/--n-embd/...) ignored; checkpoint config wins.")

    print("[2/4] Building tokenizer")
    if resumed_cfg is not None and not args.tokenizer:
        # Reuse the tokenizer that lives next to the checkpoint.
        sibling_tok = init_from.parent / "tokenizer.json"
        if sibling_tok.exists():
            tokenizer = ByteLevelBPETokenizer.load(str(sibling_tok))
            print(f"      loaded from checkpoint dir {sibling_tok} ({tokenizer})")
        else:
            raise FileNotFoundError(
                f"--init-from given but no tokenizer.json next to it ({sibling_tok}). "
                "Pass --tokenizer explicitly."
            )
    elif args.tokenizer:
        tokenizer = ByteLevelBPETokenizer.load(args.tokenizer)
        print(f"      loaded from {args.tokenizer} ({tokenizer})")
    else:
        tokenizer = ByteLevelBPETokenizer(vocab_size=args.vocab_size)
        tokenizer.train(texts, progress=True)
        print(f"      trained vocab={tokenizer.vocab}")
    # Always persist the tokenizer beside the checkpoint for a standalone dir.
    tok_path = out_dir / "tokenizer.json"
    tokenizer.save(tok_path)
    print(f"      tokenizer saved to {tok_path}")

    if resumed_cfg is not None:
        cfg = resumed_cfg
        if cfg.vocab_size != tokenizer.vocab:
            raise ValueError(
                f"checkpoint vocab {cfg.vocab_size} != tokenizer vocab {tokenizer.vocab}. "
                "Use the tokenizer that matches the checkpoint."
            )
    else:
        cfg = ModelConfig(
            vocab_size=tokenizer.vocab,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )
    print(f"      model params ~{cfg.param_count() / 1e6:.2f}M")
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    print("[3/4] Encoding dataset")
    train_data, val_data = build_dataset(texts, tokenizer, cfg.block_size, seed=args.seed)
    print(f"      train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")
    if len(train_data) < 100_000:
        print("      ⚠️ small corpus (<100k tokens): model will memorize, not generalize.")
        print("        add more .txt files to data/raw for a smarter model.")

    model = GPT(cfg).to(device)
    if resumed_cfg is not None:
        missing, unexpected = model.load_state_dict(ckpt["state_dict"])
        if missing or unexpected:
            print(f"      ⚠️ load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        print("      ✓ checkpoint weights loaded")
    print(f"      device: {device}")

    # SFT / continual runs on a small instruction set need a gentler LR so the
    # model does not forget what it learned in pre-training.
    sft_lr = args.lr
    if resumed_cfg is not None and args.lr == 3e-4:  # user did not override --lr
        sft_lr = 1e-4
        print(f"      resumed run: using gentler lr {sft_lr} (override with --lr)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=sft_lr, weight_decay=0.1)
    grad_accum = args.grad_accum

    # Cosine schedule with warmup
    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return sft_lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return sft_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    print("[4/4] Training")
    use_amp = device == "cuda" and not args.no_amp
    if use_amp:
        print("      ⚡ mixed precision (AMP) enabled - expected 1.5-2x speedup on T4")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:  # older torch (<2.3)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.train()
    step = 0
    start = time.time()
    best_val = float("inf")
    best_state = None
    stale_evals = 0
    stopped_early = False
    # Periodic checkpoint path: survives Colab disconnects mid-run.
    latest_path = out_dir / "model_latest.pt"
    while step < args.max_steps:
        optimizer.zero_grad()
        loss_accum = 0.0
        for _ in range(grad_accum):
            x, y = get_batch(train_data, cfg.block_size, args.batch_size, device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            loss_accum += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        scaler.step(optimizer)
        scaler.update()
        step += 1

        if step % args.print_every == 0 or step == args.max_steps:
            elapsed = time.time() - start
            tokens_seen = step * args.batch_size * grad_accum * cfg.block_size
            print(
                f"step {step}/{args.max_steps} | loss {loss_accum / grad_accum:.4f} "
                f"| lr {lr_at(step):.2e} | {tokens_seen / elapsed / 1000:.1f}k tok/s"
            )

        # Crash-safe periodic checkpoint: independent of eval cadence so a
        # dropped session resumes from at most --save-every steps ago.
        if step % args.save_every == 0 or step == args.max_steps:
            model.save(str(latest_path))

        if step % args.eval_every == 0 or step == args.max_steps:
            metrics = estimate_loss(model, train_data, val_data, cfg.block_size, args.batch_size, device, eval_iters=args.eval_iters)
            print(f"  eval | train loss {metrics['train']:.4f} | val loss {metrics['val']:.4f}")

            # Append to history CSV so runs are comparable over time.
            history_path = Path(args.history) if args.history else (out_dir / "history.csv")
            if step == args.eval_every or (not history_path.exists()):
                with history_path.open("w", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(["step", "train_loss", "val_loss", "lr"])
            with history_path.open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow([step, f"{metrics['train']:.4f}", f"{metrics['val']:.4f}", f"{lr_at(step):.2e}"])

            # Early stopping: keep the best weights, stop when val plateaus.
            if args.early_stop_patience > 0:
                if metrics["val"] < best_val - args.early_stop_min_delta:
                    best_val = metrics["val"]
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    stale_evals = 0
                else:
                    stale_evals += 1
                    if stale_evals >= args.early_stop_patience:
                        print(f"  🛑 early stop at step {step}: val loss not improving for {args.early_stop_patience} evals")
                        stopped_early = True
                        break

    model_path = out_dir / "model.pt"
    if best_state is not None:
        model.load_state_dict(best_state)  # restore best weights, not the overfit final ones
        print(f"      restoring best checkpoint (val loss {best_val:.4f})")
    model.save(str(model_path))
    if latest_path.exists() and latest_path != model_path:
        latest_path.unlink()  # clean up periodic checkpoint; best is in model.pt
    if stopped_early:
        print(f"Done (early-stopped). Model saved to {model_path}")
    else:
        print(f"Done. Model saved to {model_path}")
    print(f"Next: python -m aegisx.chat --model {model_path} --tokenizer {out_dir}/tokenizer.json")
    history_path = Path(args.history) if args.history else (out_dir / "history.csv")
    print(f"History: {history_path}")


if __name__ == "__main__":
    sys.exit(main())