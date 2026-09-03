import pytest

from aegisx.tokenizer import ByteLevelBPETokenizer

SAMPLE = [
    "nmap scans ports on a target host.",
    "SQL injection bypasses authentication.",
    "cross site scripting is dangerous.",
    "patch your dependencies regularly.",
    "bug bounty programs pay researchers.",
]


@pytest.fixture
def tokenizer():
    tok = ByteLevelBPETokenizer(vocab_size=512)
    tok.train(SAMPLE)
    return tok


def test_vocab_size_bounds():
    with pytest.raises(ValueError):
        ByteLevelBPETokenizer(vocab_size=100).train(SAMPLE)


def test_train_creates_merges(tokenizer):
    assert tokenizer.num_merges > 0
    assert tokenizer.vocab == 256 + 1 + tokenizer.num_merges


def test_roundtrip_known_text(tokenizer):
    text = "nmap scans ports on a target host."
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(ids) == text


def test_roundtrip_all_samples(tokenizer):
    for text in SAMPLE:
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.decode(ids) == text


def test_special_token_appended_by_default(tokenizer):
    ids = tokenizer.encode("hello")
    assert ids[-1] == tokenizer.special_id("<|endoftext|>")


def test_unicode_roundtrip(tokenizer):
    text = "Kerentanan SQL injection pada parameter login."
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(ids) == text


def test_save_load_roundtrip(tokenizer, tmp_path):
    path = tmp_path / "tok.json"
    tokenizer.save(str(path))
    loaded = ByteLevelBPETokenizer.load(str(path))
    assert loaded.vocab == tokenizer.vocab
    for text in SAMPLE:
        ids = loaded.encode(text, add_special_tokens=False)
        assert loaded.decode(ids) == text
    # Encodings must match exactly between original and loaded tokenizer.
    assert loaded.encode("nmap scans", add_special_tokens=False) == tokenizer.encode(
        "nmap scans", add_special_tokens=False
    )


def test_retrain_rejected(tokenizer):
    with pytest.raises(RuntimeError):
        tokenizer.train(["more text"])