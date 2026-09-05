"""AegisX one-shot pipeline: stage 1 (pre-train) -> stage 2 (SFT) -> stage 3 (DPO).

Runs the full training chain with a single command so you never have to run
the three Colab notebooks one by one. Everything is resume-safe:

  - Stage 1 resumes from `model_latest.pt` if a run was interrupted, and
    *archives* checkpoints from an older architecture (vocab/block mismatch)
    before training fresh.
  - Stage 2 (SFT) skips itself when a finished checkpoint matching stage 1
    exists; resumes from `model_latest.pt` otherwise.
  - Stage 3 (DPO) follows the same rule against the stage-2 checkpoint.

Usage (repo root, any machine — Colab included):

    python scripts/pipeline_full.py \
        --base-dir /content/drive/MyDrive/aegisx/checkpoints \
        --stages 1,2,3 --fetch

Run only some stages:  --stages 2,3   |   Re-run a finished stage:  --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

STAGE_DIRS = {
    1: "aegisx-mini",   # pre-train
    2: "aegisx-sft",    # fine-tune
    3: "aegisx-align",  # preference alignment
}


def run(cmd: list[str]) -> int:
    print(f"\n▶ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=False).returncode


def read_config(pt_path: Path) -> dict | None:
    """Read config.json sitting next to a checkpoint (no torch needed)."""
    cfg = pt_path.parent / "config.json"
    if not cfg.exists():
        return None
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _struct_key(cfg: dict) -> tuple:
    """Structural arch key (block/layer/head/embd), stable across tokenizers."""
    return (
        cfg.get("block_size"),
        cfg.get("n_layer"),
        cfg.get("n_head"),
        cfg.get("n_embd"),
    )


def _vocab_block_key(cfg: dict) -> tuple:
    return (cfg.get("vocab_size"), cfg.get("block_size"))


def archive_if_mismatch(out_dir: Path, want: tuple, structural: bool) -> None:
    """Move a checkpoint dir aside when its arch differs from the target.

    `structural=True` compares block/layer/head/embd only — used by stage 1
    because the real tokenizer vocab (byte base + merges) is always <= the
    requested --vocab-size, so comparing raw vocab would misfire every run.
    """
    if not out_dir.exists():
        return
    ckpt = out_dir / "model.pt"
    if not ckpt.exists():
        ckpt = out_dir / "model_latest.pt"
    if not ckpt.exists():
        return
    cfg = read_config(ckpt)
    if cfg is None:
        return
    key = _struct_key(cfg) if structural else _vocab_block_key(cfg)
    if key == want:
        return
    if structural:
        label = "-block{}".format(cfg.get("block_size"))
    else:
        label = "-vocab{}-block{}".format(cfg.get("vocab_size"), cfg.get("block_size"))
    arc = out_dir.parent / f"{out_dir.name}-archive{label}"
    if not arc.exists():
        out_dir.rename(arc)
        print(f"  ♻️  arsitektur lama {key} != target {want}; checkpoint diarsipkan ke {arc.name}")
    else:
        print(f"  ♻️  arsitektur lama {key} != target {want}; archive {arc.name} sudah ada, checkpoint dibiarkan")


def stage_1(args) -> bool:
    out = Path(args.base_dir) / STAGE_DIRS[1]
    # Bandingkan hanya arsitektur struktural; vocab aktual ditentukan tokenizer
    # (selalu <= --vocab-size karena byte-fallback BPE), jadi vocab tidak dipakai
    # sebagai penentu arsip di tahap 1.
    want = (args.block_size, args.n_layer, args.n_head, args.n_embd)
    archive_if_mismatch(out, want, structural=True)

    finished = out / "model.pt"
    if finished.exists() and not args.force:
        cfg = read_config(finished)
        if cfg and _struct_key(cfg) == want:
            print(f"  ✓ tahap 1 sudah selesai ({finished}); skip (--force untuk ulang)")
            return True

    resume = ""
    latest = out / "model_latest.pt"
    if latest.exists() and not args.force:
        cfg = read_config(latest)
        if cfg and _struct_key(cfg) == want:
            resume = str(latest)
            print(f"  ⏳ resume tahap 1 dari {latest.name}")

    cmd = [
        sys.executable, "-m", "aegisx.train",
        "--data", str(args.corpus),
        "--out", str(out),
        "--vocab-size", str(args.vocab_size),
        "--block-size", str(args.block_size),
        "--n-layer", str(args.n_layer),
        "--n-head", str(args.n_head),
        "--n-embd", str(args.n_embd),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--max-steps", str(args.max_steps),
        "--lr", str(args.lr),
        "--warmup-steps", str(args.warmup_steps),
        "--eval-every", str(args.eval_every),
        "--save-every", str(args.save_every),
        "--early-stop-patience", str(args.early_stop),
        "--device", args.device,
    ]
    if resume:
        cmd += ["--init-from", resume]
    return run(cmd) == 0


def stage_2(args) -> bool:
    base = Path(args.base_dir)
    init = base / STAGE_DIRS[1] / "model.pt"
    out = base / STAGE_DIRS[2]
    if not init.exists():
        print(f"  ✗ tahap 2 butuh {init}; jalankan tahap 1 dulu (--stages 1,2)")
        return False

    want_cfg = read_config(init)
    want = _vocab_block_key(want_cfg) if want_cfg else (0, 0)
    archive_if_mismatch(out, want, structural=False)

    finished = out / "model.pt"
    if finished.exists() and not args.force:
        print(f"  ✓ tahap 2 sudah selesai ({finished}); skip (--force untuk ulang)")
        return True

    resume = ""
    latest = out / "model_latest.pt"
    if latest.exists() and not args.force:
        cfg = read_config(latest)
        if cfg and _vocab_block_key(cfg) == want:
            resume = str(latest)
            print(f"  ⏳ resume tahap 2 dari {latest.name}")

    # Data SFT: satu direktori berisi sft_chat.txt saja.
    sft_data = base.parent / "sft-data"
    sft_data.mkdir(parents=True, exist_ok=True)
    src = REPO_ROOT / "data" / "finetune" / "sft_chat.txt"
    if not src.exists():
        print(f"  ✗ {src} belum ada; jalankan build_sft_text.py dulu")
        return False
    shutil.copy(src, sft_data / "sft.txt")

    cmd = [
        sys.executable, "-m", "aegisx.train",
        "--data", str(sft_data),
        "--out", str(out),
        "--init-from", resume or str(init),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--max-steps", str(args.sft_steps),
        "--lr", str(args.sft_lr),
        "--warmup-steps", "100",
        "--eval-every", "50",
        "--save-every", "25",
        "--early-stop-patience", "4",
        "--device", args.device,
    ]
    return run(cmd) == 0


def stage_3(args) -> bool:
    base = Path(args.base_dir)
    init = base / STAGE_DIRS[2] / "model.pt"
    out = base / STAGE_DIRS[3]
    if not init.exists():
        print(f"  ✗ tahap 3 butuh {init}; jalankan tahap 2 dulu (--stages 2,3)")
        return False

    want_cfg = read_config(init)
    want = _vocab_block_key(want_cfg) if want_cfg else (0, 0)
    archive_if_mismatch(out, want, structural=False)

    pairs = REPO_ROOT / "data" / "finetune" / "preferences.jsonl"
    if not pairs.exists():
        print(f"  ✗ {pairs} belum ada; jalankan build_preferences.py dulu")
        return False

    finished = out / "model.pt"
    if finished.exists() and not args.force:
        print(f"  ✓ tahap 3 sudah selesai ({finished}); skip (--force untuk ulang)")
        return True

    resume = ""
    latest = out / "model_latest.pt"
    if latest.exists() and not args.force:
        cfg = read_config(latest)
        if cfg and _vocab_block_key(cfg) == want:
            resume = str(latest)
            print(f"  ⏳ resume tahap 3 dari {latest.name}")

    # dpo.py mengambil --init-from: titik resume = checkpoint berkala tahap 3.
    init_for_dpo = resume or str(init)
    cmd = [
        sys.executable, "-m", "aegisx.dpo",
        "--init-from", init_for_dpo,
        "--pairs", str(pairs),
        "--out", str(out),
        "--beta", str(args.beta),
        "--batch-size", "4",
        "--grad-accum", "2",
        "--max-steps", str(args.align_steps),
        "--lr", str(args.align_lr),
        "--warmup-steps", "30",
        "--eval-every", "100",
        "--save-every", "50",
        "--early-stop-patience", "4",
        "--device", args.device,
    ]
    return run(cmd) == 0


def prep(args) -> bool:
    """Build instruction rows, SFT text, and preference pairs (idempotent)."""
    steps = [
        [sys.executable, "scripts/build_instructions.py", "--force"],
        [sys.executable, "scripts/expand_instructions.py", "--target", "1600"],
        [sys.executable, "scripts/build_sft_text.py", "--out", "data/finetune/sft_chat.txt"],
        [sys.executable, "scripts/build_preferences.py"],
    ]
    if args.fetch:
        steps.insert(0, [sys.executable, "scripts/fetch_corpus.py",
                         "--target-dir", str(args.corpus), "--cve-pages", str(args.cve_pages)])
    for cmd in steps:
        if run(cmd) != 0:
            print(f"  ✗ prep gagal: {' '.join(cmd)}")
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="AegisX pipeline: pretrain -> SFT -> DPO")
    parser.add_argument("--base-dir", default="checkpoints", help="root folder holding aegisx-mini/-sft/-align")
    parser.add_argument("--corpus", default="data/raw")
    parser.add_argument("--stages", default="1,2,3", help="comma list, e.g. 1,2,3 or 2,3")
    parser.add_argument("--fetch", action="store_true", help="jalankan fetch_corpus.py dulu (butuh internet)")
    parser.add_argument("--cve-pages", type=int, default=5, help="halaman NVD yang di-fetch (0 = skip CVE)")
    parser.add_argument("--force", action="store_true", help="ulang tahap yang sudah selesai")
    # arsitektur tahap 1
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=768)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=512)
    # training
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1500, help="tahap 1")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--sft-steps", type=int, default=200)
    parser.add_argument("--sft-lr", type=float, default=1e-4)
    parser.add_argument("--align-steps", type=int, default=600)
    parser.add_argument("--align-lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--device", default="", help="cuda/cpu (default: otomatis)")
    args = parser.parse_args()

    if not args.device:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Pipeline AegisX | device={args.device} | stages={args.stages} | base={args.base_dir}")

    stages = [int(s.strip()) for s in args.stages.split(",") if s.strip()]
    Path(args.base_dir).mkdir(parents=True, exist_ok=True)

    if not prep(args):
        sys.exit(1)

    ok = True
    t0 = time.time()
    if 1 in stages:
        ok = stage_1(args) and ok
    if 2 in stages:
        ok = stage_2(args) and ok
    if 3 in stages:
        ok = stage_3(args) and ok

    if not ok:
        print("\n✗ pipeline gagal di salah satu tahap — lihat log di atas.")
        sys.exit(1)

    base = Path(args.base_dir)
    print("\n✅ Pipeline selesai dalam %.1f menit" % ((time.time() - t0) / 60))
    for n, name in STAGE_DIRS.items():
        mp = base / name / "model.pt"
        print(f"  tahap {n} ({name}): {'✓ ' + str(mp) if mp.exists() else '— tidak ada'}")
    print("\nChat dengan model akhir (tahap 3):")
    print(f"  python -m aegisx.chat --model {base / STAGE_DIRS[3] / 'model.pt'} "
          f"--tokenizer {base / STAGE_DIRS[3] / 'tokenizer.json'}")


if __name__ == "__main__":
    sys.exit(main())