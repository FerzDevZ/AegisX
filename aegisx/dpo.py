"""Stage-3 alignment: Direct Preference Optimization (DPO-lite) for AegisX.

Takes an SFT checkpoint as the policy, freezes a copy as the reference, and
optimizes the DPO objective on (prompt, chosen, rejected) pairs so the model
prefers helpful in-scope answers and refuses out-of-scope requests — all on
your own weights (pure-zero, no third-party base model).

Loss (length-normalized DPO, beta 0.05 default):
    loss = -log sigmoid( beta * ( logp_pi(chosen) - logp_ref(chosen)
                                 - logp_pi(rejected) + logp_ref(rejected) ) )
Each sequence log-prob is normalized by its answer length so short refusals
are not unfairly favored over long helpful answers.

Usage:
    python -m aegisx.dpo --init-from <sft-model.pt> \
        --pairs data/finetune/preferences.jsonl --out checkpoints/aegisx-align
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
import torch.nn.functional as F

from aegisx.model import GPT, ModelConfig
from aegisx.tokenizer import ByteLevelBPETokenizer

BETA_DEFAULT = 0.05


def load_pairs(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def encode_pair(
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    answer: str,
    block_size: int,
    eos_id: int,
) -> tuple[list[int], list[int], int]:
    """Encode (prompt, answer) -> (ids, targets, n_answer_tokens).

    Targets are -1 for prompt positions (ignored in the loss) and the answer
    ids (plus EOS) for the answer. The prompt is truncated from the LEFT so
    the full answer always fits when the pair exceeds block_size.
    """
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_ids = tokenizer.encode(answer, add_special_tokens=False)
    a_ids = a_ids[: max(1, block_size - 2)] + [eos_id]  # fit answer fully
    budget = block_size - len(a_ids)
    p_ids = p_ids[-max(0, budget):]  # keep tail of prompt
    ids = p_ids + a_ids
    targets = [-1] * len(p_ids) + a_ids
    return ids, targets, len(a_ids)


def sequence_log_prob(model: GPT, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-sample log-prob of targets under the model (masked), shape (B,).

    Positions with target -1 (prompt tokens / padding) are excluded. We use
    log_softmax + gather instead of cross_entropy because F.cross_entropy
    rejects -1 targets.
    """
    logits, _ = model(x)  # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)
    picked = log_probs.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)  # (B, T)
    mask = (y != -1)
    if not mask.any():
        return torch.zeros(x.size(0), device=x.device)
    return (picked * mask).sum(dim=1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    ref_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_rejected: torch.Tensor,
    chosen_len: torch.Tensor,
    rejected_len: torch.Tensor,
    beta: float = BETA_DEFAULT,
) -> torch.Tensor:
    """Length-normalized DPO loss over a batch. Returns mean scalar."""
    eps = 1e-8
    lpc = policy_chosen / chosen_len.clamp_min(1)
    lrc = ref_chosen / chosen_len.clamp_min(1)
    lpr = policy_rejected / rejected_len.clamp_min(1)
    lrr = ref_rejected / rejected_len.clamp_min(1)
    logits = beta * (lpc - lrc - lpr + lrr)
    return -F.logsigmoid(logits).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO-lite alignment for AegisX")
    parser.add_argument("--init-from", required=True, help="SFT checkpoint (model.pt) to align")
    parser.add_argument("--pairs", required=True, help="preferences.jsonl (prompt/chosen/rejected)")
    parser.add_argument("--out", default="checkpoints/aegisx-align")
    parser.add_argument("--beta", type=float, default=BETA_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-5, help="DPO is fragile; keep small")
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--history", type=str, default="", help="CSV path (default <out>/history.csv)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the SFT checkpoint -> policy, plus a frozen copy -> reference.
    ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    policy = GPT(cfg).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    reference = GPT(cfg).to(device)
    reference.load_state_dict(ckpt["state_dict"])
    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()
    print(f"[init] policy/reference from {args.init_from} | {cfg}")

    # Tokenizer lives next to the checkpoint.
    tok_path = Path(args.init_from).parent / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer not found next to checkpoint: {tok_path}")
    tokenizer = ByteLevelBPETokenizer.load(str(tok_path))
    eos_id = tokenizer.special_id("<|endoftext|>")
    (out_dir / "tokenizer.json").write_text(tok_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    pairs = load_pairs(args.pairs)
    if not pairs:
        raise ValueError(f"no pairs in {args.pairs} — run scripts/build_preferences.py first")
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    split = max(1, int(len(pairs) * 0.9))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    print(f"[data] {len(pairs)} pairs ({len(train_pairs)} train / {len(val_pairs)} val)")

    def make_batch(batch_pairs: list[dict]) -> tuple[torch.Tensor, ...]:
        xs_c, ys_c, xs_r, ys_r, lens_c, lens_r = [], [], [], [], [], []
        for p in batch_pairs:
            ids_c, tgt_c, n_c = encode_pair(tokenizer, p["prompt"], p["chosen"], cfg.block_size, eos_id)
            ids_r, tgt_r, n_r = encode_pair(tokenizer, p["prompt"], p["rejected"], cfg.block_size, eos_id)
            xs_c.append(ids_c); ys_c.append(tgt_c); lens_c.append(n_c)
            xs_r.append(ids_r); ys_r.append(tgt_r); lens_r.append(n_r)
        # Pad each group to its own max length with -1 targets (ignored in loss).
        def pad(seqs, targets):
            maxlen = max(len(s) for s in seqs)
            x = torch.full((len(seqs), maxlen), 0, dtype=torch.long)
            y = torch.full((len(seqs), maxlen), -1, dtype=torch.long)
            for i, (s, t) in enumerate(zip(seqs, targets)):
                x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                y[i, : len(t)] = torch.tensor(t, dtype=torch.long)
            return x.to(device), y.to(device)
        xc, yc = pad(xs_c, ys_c)
        xr, yr = pad(xs_r, ys_r)
        return xc, yc, xr, yr, torch.tensor(lens_c, device=device), torch.tensor(lens_r, device=device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    use_amp = device == "cuda" and not args.no_amp
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def eval_margin() -> dict[str, float]:
        """Mean (chosen_len-norm logp) - (rejected_len-norm logp) over val."""
        policy.eval()
        margins = []
        with torch.no_grad():
            for i in range(0, len(val_pairs), args.batch_size):
                xc, yc, xr, yr, lc, lr = make_batch(val_pairs[i : i + args.batch_size])
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    lpc = sequence_log_prob(policy, xc, yc) / lc.clamp_min(1)
                    lpr = sequence_log_prob(policy, xr, yr) / lr.clamp_min(1)
                margins.append((lpc - lpr).cpu())
        policy.train()
        m = torch.cat(margins) if margins else torch.tensor([])
        return {"margin": float(m.mean()) if len(m) else 0.0, "n": len(m)}

    best_margin = -float("inf")
    best_state = None
    stale = 0
    step = 0
    start = time.time()
    latest_path = out_dir / "model_latest.pt"
    policy.train()
    while step < args.max_steps:
        optimizer.zero_grad()
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            batch = rng.sample(train_pairs, min(args.batch_size, len(train_pairs)))
            xc, yc, xr, yr, lc, lr = make_batch(batch)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                with torch.no_grad():
                    ref_c = sequence_log_prob(reference, xc, yc)
                    ref_r = sequence_log_prob(reference, xr, yr)
                pol_c = sequence_log_prob(policy, xc, yc)
                pol_r = sequence_log_prob(policy, xr, yr)
                loss = dpo_loss(pol_c, ref_c, pol_r, ref_r, lc, lr, beta=args.beta)
            scaler.scale(loss).backward()
            loss_accum += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        scaler.step(optimizer)
        scaler.update()
        step += 1

        if step % 10 == 0 or step == args.max_steps:
            elapsed = time.time() - start
            print(f"step {step}/{args.max_steps} | dpo {loss_accum / args.grad_accum:.4f} | lr {lr_at(step):.2e} | {step / elapsed:.1f} it/s")

        if step % args.save_every == 0:
            policy.save(str(latest_path))

        if step % args.eval_every == 0 or step == args.max_steps:
            metrics = eval_margin()
            print(f"  eval | val margin {metrics['margin']:.4f} (n={metrics['n']})")
            history_path = Path(args.history) if args.history else (out_dir / "history.csv")
            header = not history_path.exists()
            with history_path.open("a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if header:
                    w.writerow(["step", "margin", "lr"])
                w.writerow([step, f"{metrics['margin']:.4f}", f"{lr_at(step):.2e}"])
            if metrics["margin"] > best_margin + 1e-4:
                best_margin = metrics["margin"]
                best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if args.early_stop_patience > 0 and stale >= args.early_stop_patience:
                    print(f"  🛑 early stop at step {step}: margin not improving for {args.early_stop_patience} evals")
                    break

    model_path = out_dir / "model.pt"
    if best_state is not None:
        policy.load_state_dict(best_state)
        print(f"      restoring best checkpoint (margin {best_margin:.4f})")
    policy.save(str(model_path))
    if latest_path.exists() and latest_path != model_path:
        latest_path.unlink()
    print(f"Done. Model saved to {model_path}")
    print(f"Next: python -m aegisx.chat --model {model_path} --tokenizer {out_dir}/tokenizer.json")


if __name__ == "__main__":
    sys.exit(main())