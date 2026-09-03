# ⚡ AegisX-Mini

A personal cybersecurity AI model — **trained from scratch**, light enough for a
"potato" laptop, designed to grow into a full bug-bounty assistant over time.

| | |
|---|---|
| **Model** | GPT-style decoder-only transformer, 4–50M params |
| **Tokenizer** | Byte-level BPE (dependency-free, trains on your corpus) |
| **Training** | Pure PyTorch, CPU-friendly, on your laptop **or** free Colab T4 |
| **Hosting** | Hugging Face Hub + Space (Gradio chat) |
| **Agent layer** | Recon/scan tools behind an authorization gate + audit log |
| **RAG** | Zero-dependency retrieval to ground answers with sources |

> Full design in [RFC.md](RFC.md). Status: **v0.1 — MVP training pipeline working.**

---

## Quickstart (laptop, CPU)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train AegisX-Mini from scratch on the bundled seed corpus
python -m aegisx.train --data data/raw --out checkpoints/aegisx-mini \
    --vocab-size 4096 --block-size 128 --n-layer 4 --n-head 4 --n-embd 256 \
    --max-steps 2000

# 3. Chat with it
python -m aegisx.chat --model checkpoints/aegisx-mini/model.pt \
    --tokenizer checkpoints/aegisx-mini/tokenizer.json
```

Add your own corpus: drop `.txt` files into `data/raw/` (one topic per file).
The model learns from whatever you feed it — quality in, quality out.

---

## Project layout

```
aegisx/
  tokenizer.py    # byte-level BPE: train / encode / decode / save / load
  model.py        # GPT decoder-only transformer (config, generate, save/load)
  train.py        # CPU-friendly training: dataset, tokenizer, checkpointing
  chat.py         # interactive chat + one-shot generation
  gate.py         # authorization gate: allowlist + audit log + safe exec
  agent.py        # agent tools (recon, web, code search, reports) — gated
data/raw/         # seed corpus (.txt). Add your own.
targets/          # authorized targets allowlist (edit me!)
notebooks/        # Colab training notebook (free T4)
hf/               # Hugging Face: push script + Space app
tests/            # pytest suite (tokenizer, model, training, gate)
```

---

## Verify

```bash
pytest -q
```

Expected: all green (59 tests). The suite covers the tokenizer round-trip,
model forward/training/generation, dataset splitting, early-stop end-to-end
runs, tokenizer resume, chat one-shot, the authorization gate (allowlist,
audit log, sudo/shell injection blocking), RAG retrieval, and agent-CLI tool
calls.

## Grow the corpus

```bash
# Pull public cysec corpora (OWASP sheets, ASVS, MITRE ATT&CK, CVE records)
python3 scripts/fetch_corpus.py --target-dir data/raw
```

Then retrain: `python3 -m aegisx.train --data data/raw --out checkpoints/aegisx-mini`

## Ground answers (RAG)

```python
from aegisx.rag import CorpusIndex

idx = CorpusIndex()
idx.add_dir('data/raw')
for chunk in idx.retrieve('sql injection prevention', top_k=3):
    print(f'[{chunk.source}] {chunk.text[:200]}')
```

## Agent mode (gated tool calls)

```bash
python -m aegisx.agent_cli --model checkpoints/aegisx-mini/model.pt \
    --tokenizer checkpoints/aegisx-mini/tokenizer.json
# then type: @tool recon_ports example.com  (gated by targets/authorized.txt)
```

---

## Agent layer (bug-bounty mode)

All offensive actions go through `AuthorizationGate`, which enforces:

1. **Target allowlist** — edit `targets/authorized.txt`; anything not listed is **denied**.
2. **Audit log** — every action is timestamped and appended to `logs/audit.log`.
3. **Safe execution** — `sudo` and shell metacharacters are blocked; commands run as `argv` only.

```python
from aegisx.gate import AuthorizationGate
from aegisx.agent import AegisAgent

gate = AuthorizationGate("targets/authorized.txt", "logs/audit.log")
agent = AegisAgent(gate)

print(agent.recon_ports("192.168.1.10", ports="22,80,443"))
print(agent.write_report(
    "192.168.1.10", "SQL Injection in login", "High",
    "Parameterized queries not used", "curl -X POST ...", "Full DB read"
))
```

> ⚠️ Only ever run this against hosts **you own or have written permission to
> test**. Bug bounty = in-scope assets only.

---

## Roadmap

- [x] v0.1 — tokenizer, model, training, chat, gate (this repo)
- [ ] v0.2 — Colab notebook → Hugging Face Hub push → public Gradio Space
- [ ] v0.3 — RAG knowledge base (CVE + OWASP + playbooks)
- [ ] v0.4 — agent ↔ model wiring (AegisX picks the tool, gate enforces scope)
- [ ] v0.5 — fine-tuned AegisX (QLoRA on Colab) for real bug-bounty power

See [RFC.md](RFC.md) for the full plan.