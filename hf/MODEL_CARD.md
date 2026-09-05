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
  - pytorch
  - from-scratch
pipeline_tag: text-generation
inference:
  parameters:
    temperature: 0.8
    top_k: 50
    repetition_penalty: 1.1
---

# ⚡ AegisX-Mini

A lightweight GPT-style decoder-only language model trained **from scratch** on
public cybersecurity text (English + Bahasa Indonesia). Built with the AegisX
training repo: [github.com/FerzDevZ/AegisX](https://github.com/FerzDevZ/AegisX).

AegisX-Mini is intentionally small so it can be trained and served on free
hardware (Colab T4 / CPU Spaces). It is a **starting point** for learning about
LLM training and for a focused cybersecurity assistant, not a replacement for
large frontier models.

## Files

| File             | Description                                        |
| ---------------- | -------------------------------------------------- |
| `model.pt`       | PyTorch weights (load with `aegisx.model.GPT.load`)|
| `tokenizer.json` | Byte-level BPE tokenizer (required)                |
| `config.json`    | Architecture config                                |

## Architecture

| Hyperparameter | Value  |
| -------------- | ------ |
| Params         | ~12–30M (see `config.json`) |
| Layers         | 6–8    |
| Heads          | 6–8    |
| Embed dim      | 384–512|
| Context length | 256 tokens |
| Vocab          | ~3–4K (trained BPE) |

## Training data

Public cybersecurity corpus, English + Indonesian:

- OWASP Cheat Sheet Series, OWASP ASVS, MITRE ATT&CK (techniques)
- Sample CVE descriptions
- Curated topic files: pentest methodology & tools, OSINT, mobile/API
  security, report writing, crypto & network security, defense
- Bahasa Indonesia Q&A in `User: ... AegisX: ...` chat format

## Use it

```bash
pip install -r requirements.txt
python -m aegisx.chat \
    --model model.pt \
    --tokenizer tokenizer.json \
    --prompt "Apa itu SQL injection?\n\nAegisX:"
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

## Limitations

- **Small model + small corpus**: can parrot training text, struggle with
  novel questions, and may not be fluent in Indonesian yet.
- Do not rely on it for production security decisions without human review.
- Keep usage **authorized**: only test systems you own or have permission to
  test.

## License

MIT — free to use, fine-tune, and redistribute.
