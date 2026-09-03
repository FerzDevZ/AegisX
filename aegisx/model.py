"""AegisX-Mini: a small GPT-style decoder-only transformer.

Pure PyTorch, runs on CPU. Sized for a "potato" laptop (4-50M params).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    block_size: int = 256        # max context length (tokens)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False           # no bias in LayerNorm/QKV for simplicity
    tie_weights: bool = True     # tie token embedding & lm_head (saves params)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f: d[f] for f in asdict(cls()) if f in d}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)

    def param_count(self) -> int:
        return param_count_approx(self)


def param_count_approx(cfg: ModelConfig) -> int:
    """Closed-form parameter estimate (embedding ties included)."""
    n_embd, n_head, n_layer, vocab = cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.vocab_size
    head_size = n_embd // n_head
    del head_size
    # per layer: attention (c_attn 3*n_embd*n_embd + c_proj) + MLP (up 4x + down)
    attn = 3 * n_embd * n_embd + n_embd * n_embd
    mlp = 2 * n_embd * (4 * n_embd)
    per_layer = attn + mlp
    total = n_layer * per_layer
    if cfg.tie_weights:
        total += vocab * n_embd  # embedding counted once (tied to lm_head)
    else:
        total += 2 * vocab * n_embd
    total += n_embd  # final layernorm gamma
    return total


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.config = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=LayerNorm(cfg.n_embd, bias=cfg.bias),
            )
        )
        if cfg.tie_weights:
            self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.transformer.wte.weight  # weight tying
        else:
            self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.config.block_size, f"seq len {T} > block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = self._masked_cross_entropy(logits, targets)
        return logits, loss

    @staticmethod
    def _masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross entropy over non-ignored positions; 0.0 if all are ignored."""
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)
        mask = flat_targets != -1
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return F.cross_entropy(flat_logits[mask], flat_targets[mask])

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Generate tokens continuing from idx (B, T)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if repetition_penalty != 1.0:
                prev = idx[:, -self.config.block_size :]
                for tok in torch.unique(prev[0]):
                    logits[0, tok] = logits[0, tok] / repetition_penalty if logits[0, tok] > 0 else logits[0, tok] * repetition_penalty

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def save(self, path: str) -> None:
        torch.save(
            {"config": self.config.to_dict(), "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "GPT":
        data = torch.load(path, map_location=device, weights_only=False)
        cfg = ModelConfig.from_dict(data["config"])
        model = cls(cfg)
        model.load_state_dict(data["state_dict"])
        model.to(device)
        return model