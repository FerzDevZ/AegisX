import pytest
import torch

from aegisx.dpo import dpo_loss, sequence_log_prob
from aegisx.preferences import build_pairs, write_pairs_jsonl


def test_dpo_loss_prefers_chosen():
    """Loss kecil saat chosen lebih mungkin daripada rejected (margin positif)."""
    beta = 0.05
    # chosen jauh lebih baik -> margin besar -> loss ~ -log sigmoid(big) ~ 0
    good = dpo_loss(
        torch.tensor([-1.0]), torch.tensor([-1.5]),
        torch.tensor([-9.0]), torch.tensor([-1.5]),
        torch.tensor([10.0]), torch.tensor([10.0]),
        beta=beta,
    )
    # rejected lebih baik -> margin negatif -> loss lebih besar
    bad = dpo_loss(
        torch.tensor([-9.0]), torch.tensor([-1.5]),
        torch.tensor([-1.0]), torch.tensor([-1.5]),
        torch.tensor([10.0]), torch.tensor([10.0]),
        beta=beta,
    )
    assert good.item() < bad.item()
    assert good.item() < 1.0


def test_dpo_loss_margin_zero_when_equal():
    loss = dpo_loss(
        torch.tensor([-2.0]), torch.tensor([-2.0]),
        torch.tensor([-2.0]), torch.tensor([-2.0]),
        torch.tensor([5.0]), torch.tensor([5.0]),
        beta=0.05,
    )
    assert abs(loss.item() - 0.6931) < 1e-2  # -log sigmoid(0)


def test_dpo_loss_length_normalization():
    """Jawaban panjang vs pendek dengan rata-rata logprob sama => margin setara."""
    l1 = dpo_loss(
        torch.tensor([-20.0]), torch.tensor([-20.0]),
        torch.tensor([-10.0]), torch.tensor([-10.0]),
        torch.tensor([10.0]), torch.tensor([5.0]),
        beta=0.05,
    )
    l2 = dpo_loss(
        torch.tensor([-2.0]), torch.tensor([-2.0]),
        torch.tensor([-1.0]), torch.tensor([-1.0]),
        torch.tensor([1.0]), torch.tensor([1.0]),
        beta=0.05,
    )
    assert abs(l1.item() - l2.item()) < 1e-3


def test_sequence_log_prob_matches_manual():
    from aegisx.model import GPT, ModelConfig

    cfg = ModelConfig(vocab_size=64, block_size=16, n_layer=1, n_head=1, n_embd=16)
    model = GPT(cfg).eval()
    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[-1, -1, 5, 6]])  # target hanya 2 posisi terakhir
    with torch.no_grad():
        logits, _ = model(x)
    logp = torch.log_softmax(logits, dim=-1)
    manual = logp[0, 2, 5].item() + logp[0, 3, 6].item()
    auto = sequence_log_prob(model, x, y)[0].item()
    assert abs(manual - auto) < 1e-4


def test_build_pairs_shapes_and_refusals():
    rows = [
        {"instruction": "Apa itu SQL injection?", "output": "SQL injection adalah ..."},
        {"instruction": "Bagaimana cara memindai port?", "output": "Gunakan nmap ..."},
    ]
    pairs = build_pairs(rows, seed=7)
    assert len(pairs) == len(rows) + 15  # 2 in-scope + 15 out-of-scope templates
    for p in pairs:
        assert set(p) == {"prompt", "chosen", "rejected"}
        assert p["prompt"] and p["chosen"] and p["rejected"]
    # Pasti ada pasangan yang chosen-nya penolakan (out-of-scope).
    assert any("tidak bisa" in p["chosen"] or "tidak akan" in p["chosen"] for p in pairs)


def test_write_pairs_jsonl(tmp_path):
    out = tmp_path / "preferences.jsonl"
    pairs = build_pairs(
        [{"instruction": "q", "output": "a"}], seed=1
    )
    n = write_pairs_jsonl(pairs, out)
    assert n == len(pairs)
    assert out.exists()
    import json

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(pairs)
    first = json.loads(lines[0])
    assert "prompt" in first and "chosen" in first and "rejected" in first