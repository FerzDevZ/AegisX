import pytest

from aegisx.train import build_dataset, load_texts
from aegisx.tokenizer import ByteLevelBPETokenizer

SAMPLE = [
    "nmap scans ports on a target host.",
    "SQL injection bypasses authentication.",
    "cross site scripting is dangerous.",
    "patch your dependencies regularly.",
    "bug bounty programs pay researchers.",
    "defense in depth layers your controls.",
    "encrypt data in transit and at rest.",
    "incident response contains the breach.",
    "phishing attacks target the human.",
    "least privilege reduces the damage.",
    "hardening shrinks the attack surface.",
    "monitoring detects attacks in progress.",
    "patching fixes known vulnerabilities.",
    "segmentation contains the compromise.",
    "backups must be tested and isolated.",
    "authentication is the first line.",
    "zero trust verifies every request.",
    "reconnaissance maps the target.",
    "fingerprinting identifies the service.",
    "exploitation gains access to the system.",
    "report writing is a skill.",
    "scope defines what you may test.",
    "duplicates get no bounty.",
    "business logic bugs need reasoning.",
    "race conditions need concurrency testing.",
    "idor exposes other users data.",
    "xss injects client side scripts.",
    "ssrf reaches internal services.",
    "xxe reads local files.",
    "cve identifiers track vulnerabilities.",
]


def test_load_texts_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_texts(str(tmp_path))


def test_load_texts_reads_txt(tmp_path):
    (tmp_path / "a.txt").write_text(
        "This is the first paragraph with enough length to pass the minimum threshold.\n\n"
        "This is the second paragraph with enough length to pass the threshold too.\n",
        encoding="utf-8",
    )
    texts = load_texts(str(tmp_path))
    assert len(texts) == 2
    assert texts[0].startswith("This is the first paragraph")


def test_load_texts_skips_short_fragments(tmp_path):
    (tmp_path / "a.txt").write_text("tiny\n\nnormal length paragraph for testing purposes.\n", encoding="utf-8")
    texts = load_texts(str(tmp_path))
    assert all(len(t) >= 32 for t in texts)
    assert len(texts) == 1


def test_build_dataset_splits_train_val():
    tokenizer = ByteLevelBPETokenizer(vocab_size=512)
    tokenizer.train(SAMPLE)
    train, val = build_dataset(SAMPLE, tokenizer, block_size=16, seed=7)
    assert len(train) > 0
    assert len(val) > 0
    assert len(train) + len(val) == sum(len(tokenizer.encode(t, add_special_tokens=False)) + 1 for t in SAMPLE)


def test_build_dataset_contains_eos():
    tokenizer = ByteLevelBPETokenizer(vocab_size=512)
    tokenizer.train(SAMPLE)
    eos = tokenizer.special_id("<|endoftext|>")
    train, _ = build_dataset(SAMPLE, tokenizer, block_size=16, seed=7)
    assert eos in train.tolist()


def test_get_batch_tiny_dataset_does_not_crash():
    """Regression: val set smaller than block_size used to crash randint."""
    import torch

    from aegisx.train import get_batch

    # 223 tokens < block_size 256 — the exact Colab failure.
    data = torch.arange(223, dtype=torch.long)
    x, y = get_batch(data, block_size=256, batch_size=4, device="cpu")
    assert x.shape == (1, 222)
    assert y.shape == (1, 222)
    assert torch.equal(y[0], x[0] + 1)


def test_get_batch_normal_path():
    import torch

    from aegisx.train import get_batch

    data = torch.arange(1000, dtype=torch.long)
    x, y = get_batch(data, block_size=64, batch_size=8, device="cpu")
    assert x.shape == (8, 64)
    assert y.shape == (8, 64)
    assert torch.equal(y, x + 1)


def _write_small_corpus(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "a.txt").write_text("\n\n".join(SAMPLE), encoding="utf-8")
    return tmp_path


def test_end_to_end_tiny_train_saves_model(tmp_path):
    """Q1: tiny corpus + early stopping completes and writes model.pt."""
    import subprocess
    import sys

    data_dir = _write_small_corpus(tmp_path / "data")
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable, "-m", "aegisx.train",
            "--data", str(data_dir), "--out", str(out),
            "--vocab-size", "512", "--block-size", "32",
            "--n-layer", "1", "--n-head", "1", "--n-embd", "32",
            "--batch-size", "2", "--grad-accum", "1",
            "--max-steps", "30", "--eval-every", "5", "--eval-iters", "2",
            "--early-stop-patience", "2", "--warmup-steps", "5",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "model.pt").exists()
    assert (out / "tokenizer.json").exists()
    assert (out / "config.json").exists()


def test_save_every_writes_periodic_checkpoint_before_eval(tmp_path):
    """--save-every persists model_latest.pt mid-run, before any eval fires.

    This is the crash-safety mechanism: if the Colab session dies, the
    periodic checkpoint already exists so the next run resumes from it.
    """
    import subprocess
    import sys
    import time

    data_dir = _write_small_corpus(tmp_path / "data")
    out = tmp_path / "out"
    # eval-every huge => no eval would ever fire; only --save-every writes.
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "aegisx.train",
            "--data", str(data_dir), "--out", str(out),
            "--vocab-size", "512", "--block-size", "32",
            "--n-layer", "1", "--n-head", "1", "--n-embd", "32",
            "--batch-size", "2", "--grad-accum", "1",
            "--max-steps", "100000", "--eval-every", "100000", "--eval-iters", "2",
            "--save-every", "5", "--print-every", "100000", "--warmup-steps", "3",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    latest = out / "model_latest.pt"
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            if latest.exists() and latest.stat().st_size > 0:
                break
            time.sleep(0.5)
        assert latest.exists(), "model_latest.pt was never written (--save-every broken?)"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_train_resume_with_existing_tokenizer(tmp_path):
    """Q2: --tokenizer resume path reuses a saved tokenizer."""
    import subprocess
    import sys

    data_dir = _write_small_corpus(tmp_path / "data")
    out1 = tmp_path / "out1"
    subprocess.run(
        [
            sys.executable, "-m", "aegisx.train",
            "--data", str(data_dir), "--out", str(out1),
            "--vocab-size", "512", "--block-size", "32",
            "--n-layer", "1", "--n-head", "1", "--n-embd", "32",
            "--batch-size", "2", "--grad-accum", "1",
            "--max-steps", "5", "--eval-every", "10", "--eval-iters", "2",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    out2 = tmp_path / "out2"
    proc = subprocess.run(
        [
            sys.executable, "-m", "aegisx.train",
            "--data", str(data_dir), "--out", str(out2),
            "--tokenizer", str(out1 / "tokenizer.json"),
            "--vocab-size", "512", "--block-size", "32",
            "--n-layer", "1", "--n-head", "1", "--n-embd", "32",
            "--batch-size", "2", "--grad-accum", "1",
            "--max-steps", "5", "--eval-every", "10", "--eval-iters", "2",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    # Resume run must NOT retrain a new tokenizer from scratch - reuse the file.
    import json
    tok1 = json.loads((out1 / "tokenizer.json").read_text())
    tok2 = json.loads((out2 / "tokenizer.json").read_text())
    assert tok1 == tok2


def test_chat_one_shot_returns_text(tmp_path):
    """Q3: chat generate() one-shot mode returns a non-empty string."""
    import subprocess
    import sys

    data_dir = _write_small_corpus(tmp_path / "data")
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable, "-m", "aegisx.train",
            "--data", str(data_dir), "--out", str(out),
            "--vocab-size", "512", "--block-size", "32",
            "--n-layer", "1", "--n-head", "1", "--n-embd", "32",
            "--batch-size", "2", "--grad-accum", "1",
            "--max-steps", "5", "--eval-every", "10", "--eval-iters", "2",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    proc = subprocess.run(
        [
            sys.executable, "-m", "aegisx.chat",
            "--model", str(out / "model.pt"),
            "--tokenizer", str(out / "tokenizer.json"),
            "--prompt", "hello", "--max-new-tokens", "10",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip()


def test_train_init_from_resumes_checkpoint(tmp_path):
    """Stage-2 SFT: --init-from loads a previous model.pt and continues."""
    import subprocess
    import sys

    data_dir = _write_small_corpus(tmp_path / "data")
    out1 = tmp_path / "stage1"
    # Stage 1: tiny pre-train.
    proc1 = subprocess.run(
        [
            sys.executable, "-m", "aegisx.train",
            "--data", str(data_dir), "--out", str(out1),
            "--vocab-size", "512", "--block-size", "32",
            "--n-layer", "1", "--n-head", "1", "--n-embd", "32",
            "--batch-size", "2", "--grad-accum", "1",
            "--max-steps", "15", "--eval-every", "5", "--eval-iters", "2",
            "--warmup-steps", "3",
        ],
        capture_output=True,
        text=True,
    )
    assert proc1.returncode == 0, proc1.stderr
    assert (out1 / "model.pt").exists()
    assert (out1 / "tokenizer.json").exists()

    # Stage 2: SFT resuming stage-1 weights with --init-from (no arch args).
    out2 = tmp_path / "stage2"
    proc2 = subprocess.run(
        [
            sys.executable, "-m", "aegisx.train",
            "--data", str(data_dir), "--out", str(out2),
            "--init-from", str(out1 / "model.pt"),
            "--max-steps", "10", "--eval-every", "5", "--eval-iters", "2",
        ],
        capture_output=True,
        text=True,
    )
    assert proc2.returncode == 0, proc2.stderr
    assert (out2 / "model.pt").exists()
    assert "checkpoint weights loaded" in proc2.stdout
    # Tokenizer must be reused from the checkpoint dir, not retrained.
    import json
    tok1 = json.loads((out1 / "tokenizer.json").read_text())
    tok2 = json.loads((out2 / "tokenizer.json").read_text())
    assert tok1 == tok2

    # Stage 2 config must equal stage-1 config (arch comes from checkpoint).
    import json as _json
    c1 = _json.loads((out1 / "config.json").read_text())
    c2 = _json.loads((out2 / "config.json").read_text())
    assert c1 == c2


def test_init_from_missing_checkpoint_fails(tmp_path):
    """A missing --init-from path must error loudly, not train from scratch."""
    import subprocess
    import sys

    data_dir = _write_small_corpus(tmp_path / "data")
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable, "-m", "aegisx.train",
            "--data", str(data_dir), "--out", str(out),
            "--init-from", str(tmp_path / "nope.pt"),
            "--max-steps", "5",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "not found" in proc.stderr


def test_build_dataset_seeded():
    tokenizer = ByteLevelBPETokenizer(vocab_size=512)
    tokenizer.train(SAMPLE)
    t1, v1 = build_dataset(SAMPLE, tokenizer, block_size=16, seed=3)
    t2, v2 = build_dataset(SAMPLE, tokenizer, block_size=16, seed=3)
    assert torch_equal(t1, t2)
    assert torch_equal(v1, v2)


def torch_equal(a, b) -> bool:
    return bool((a == b).all())