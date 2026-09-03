import torch
import pytest

from aegisx.model import GPT, ModelConfig, param_count_approx


def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=512,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )


def test_forward_shape():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, None)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is None


def test_loss_computed_with_targets():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, y)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0.0  # untrained model should have high loss


def test_ignore_index_skips():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.full((2, 16), -1, dtype=torch.long)
    _, loss = model(x, y)
    assert loss.item() == 0.0


def test_training_reduces_loss():
    cfg = tiny_cfg()
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, cfg.vocab_size, (4, 16))
    y = torch.randint(0, cfg.vocab_size, (4, 16))
    model.train()
    _, loss_before = model(x, y)
    for _ in range(20):
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
    _, loss_after = model(x, y)
    assert loss_after.item() < loss_before.item()


def test_generate_produces_tokens():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = model.generate(x, max_new_tokens=10, temperature=0.8, top_k=10)
    assert out.shape == (1, 13)
    assert out[0, :3].tolist() == [1, 2, 3]


def test_generate_never_exceeds_block_size():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    out = model.generate(x, max_new_tokens=50)
    assert out.shape[1] == cfg.block_size + 50


def test_save_load_roundtrip(tmp_path):
    cfg = tiny_cfg()
    model = GPT(cfg)
    path = str(tmp_path / "model.pt")
    model.save(path)
    loaded = GPT.load(path)
    assert loaded.config == cfg
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        l1, _ = model(x)
        l2, _ = loaded(x)
    assert torch.equal(l1, l2)


def test_param_count_approx_reasonable():
    cfg = ModelConfig(vocab_size=4096, block_size=128, n_layer=4, n_head=4, n_embd=256)
    n = param_count_approx(cfg)
    assert 1_000_000 < n < 100_000_000  # 4-50M ballpark for our sizes


def test_tied_weights():
    cfg = tiny_cfg()
    model = GPT(cfg)
    assert model.lm_head.weight is model.transformer.wte.weight