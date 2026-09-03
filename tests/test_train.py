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


def test_build_dataset_seeded():
    tokenizer = ByteLevelBPETokenizer(vocab_size=512)
    tokenizer.train(SAMPLE)
    t1, v1 = build_dataset(SAMPLE, tokenizer, block_size=16, seed=3)
    t2, v2 = build_dataset(SAMPLE, tokenizer, block_size=16, seed=3)
    assert torch_equal(t1, t2)
    assert torch_equal(v1, v2)


def torch_equal(a, b) -> bool:
    return bool((a == b).all())