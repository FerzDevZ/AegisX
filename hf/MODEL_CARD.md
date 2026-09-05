---
license: mit
language:
  - en
  - id
tags:
  - cybersecurity
  - penetration-testing
  - bug-bounty
  - osint
  - defensive-security
  - digital-forensics
  - pytorch
  - from-scratch
  - alignment
  - dpo
pipeline_tag: text-generation
inference:
  parameters:
    temperature: 0.8
    top_k: 50
    repetition_penalty: 1.1
---

# ⚡ AegisX-Mini

A lightweight GPT-style decoder-only language model trained **from scratch**
(pure zero — no third-party base weights) on public cybersecurity text in
**English + Bahasa Indonesia**, then aligned in 3 stages: pre-training →
SFT (`User:/AegisX:` chat format) → DPO preference alignment (refuses
off-scope / harmful requests in Indonesian).

Built with the AegisX training repo:
[github.com/FerzDevZ/AegisX](https://github.com/FerzDevZ/AegisX)

AegisX-Mini is intentionally small so it can be trained and served on free
hardware (Colab T4 / CPU Spaces). It is a **starting point** for learning LLM
training and for a focused cybersecurity assistant — **not** a replacement for
large frontier models.

## Files

| File | Description |
| ---- | ----------- |
| `model.pt` | PyTorch weights (load with `aegisx.model.GPT.load`) |
| `tokenizer.json` | Byte-level BPE tokenizer trained on the corpus (required) |
| `config.json` | Architecture config (source of truth for this run) |
| `knowledge/` | RAG grounding corpus used by the Space app (optional for CLI) |
| `README.md` | This model card |

## Architecture

| Hyperparameter | Default (v0.7) |
| -------------- | -------------- |
| Params | ~30M (see `config.json`) |
| Layers | 8 |
| Heads | 8 |
| Embed dim | 512 |
| Context length | 768 tokens |
| Vocab | 8192 (trained byte-level BPE) |

Actual values always live in `config.json` next to the weights.

## Training

Three-stage pipeline, all weights trained from random init on free Colab/Kaggle:

1. **Pre-training** — next-token prediction over the raw corpus; early stop on
   validation loss plateau with best-weight restore; AMP on GPU; crash-safe
   periodic checkpoints (`model_latest.pt`).
2. **SFT** — teaches the `User: … AegisX: …` chat format from 1,371 instruction
   rows (80% Indonesian).
3. **DPO alignment** — preference pairs (chosen/rejected, 1,386 rows, mostly
   Indonesian, β=0.05) so the model follows the user, stays on topic, and
   politely refuses harmful or unauthorized requests.

## Training data

Public, properly-licensed cybersecurity corpus, English + Indonesian (~46 MB):

- OWASP (Cheat Sheet Series, ASVS, WSTG, API-Security, Top10, MASTG)
- MITRE ATT&CK (enterprise, ICS, mobile)
- HackTricks (CC BY-NC 4.0, attribution), PayloadsAllTheThings
- CVE descriptions (NVD bulk)
- Curated topic files: pentest methodology & tools, OSINT, mobile/API
  security, report writing, crypto & network security, defense, digital
  forensics, malware analysis, wireless
- Indonesian: Wikipedia (cybersecurity category) + chat-style lessons in
  `User: … AegisX: …` format

## Use it

```bash
pip install -r requirements.txt
python -m aegisx.chat \
    --model model.pt \
    --tokenizer tokenizer.json \
    --prompt "Apa itu SQL injection?\n\nAegisX:"
```

Evaluate on the fixed 20-question set:

```bash
python -m aegisx.eval --model model.pt --tokenizer tokenizer.json
```

Agentic chat (ReAct, gated tools, human confirmation):

```bash
python -m aegisx.agent_cli --model model.pt --tokenizer tokenizer.json \
    --allowlist targets/authorized.txt --confirm
```

Or in Python:

```python
from aegisx.model import GPT
from aegisx.tokenizer import ByteLevelBPETokenizer

model = GPT.load("model.pt", device="cpu")
tok = ByteLevelBPETokenizer.load("tokenizer.json")
ids = tok.encode("You are AegisX. User: how do I enumerate subdomains?\n\nAegisX:", add_special_tokens=True)
# ... generate with model.generate(...) — see aegisx/chat.py
```

RAG grounding (zero-dependency BM25 index, source-aware chunking):

```python
from aegisx.rag import CorpusIndex

idx = CorpusIndex()
idx.add_dir("knowledge")
for chunk in idx.retrieve("sql injection prevention", top_k=3):
    print(f"[{chunk.source}] {chunk.text[:200]}")
```

## Limitations

- **Small model + growing corpus**: can parrot training text, struggle with
  novel questions, and Indonesian fluency is still developing (corpus is ~3%
  Indonesian; SFT instructions are ~80% Indonesian).
- Do not rely on it for production security decisions without human review.
- Keep usage **authorized**: only test systems you own or have permission to
  test. The model is trained to refuse unauthorized requests — if it does not,
  treat that as a red-team finding for the next alignment round.

## License

MIT — free to use, fine-tune, and redistribute.
