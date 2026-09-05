# 🛡️ AegisX-Mini — Development Plan (Master)

> Compiled by `@planner` + all sub-agents · Status: **development phase**
> Companion docs: `RFC.md` (architecture), `README.md` (usage)
>
> The single truth: **the model's power comes from its data.** All other
> improvements amplify that data. Priority order below reflects this.

---

## 0. Where we are (audit by `@qa`)

| Area | State |
|---|---|
| Pipeline | ✅ Train → chat → export works end-to-end (verified on CPU + Colab) |
| Tests | ✅ 77/77 passing (tokenizer, model, training, gate, RAG, agent CLI, eval, clean) |
| Corpus | ✅ **26 MB / 194 files** — OWASP (ASVS + CheatSheetSeries penuh ~120 sheet), MITRE ATT&CK (enterprise + ICS + mobile), WSTG, PayloadsAllTheThings, CVE (NVD bulk ~38k), custom EN+ID |
| Early stopping | ✅ stops at val-loss plateau, restores best weights |
| AMP (T4) | ✅ mixed precision on CUDA (`--no-amp` to disable), ~1.5-2x faster |
| History CSV | ✅ `history.csv` logs every eval; `aegisx.eval --history` appends scores per run |
| Corpus cleaner | ✅ `scripts/clean_corpus.py` dedup + strip noise (`--apply`, .bak backup) |
| Periodic checkpoint | ✅ `model_latest.pt` every eval (crash-safe long Colab runs) |
| RAG | ✅ zero-dep `aegisx/rag.py`: chunk + retrieve with source (df-cache for fast builds) |
| Eval | ✅ `aegisx/eval.py`: 20 fixed Q (10 EN + 10 ID), keyword-coverage scoring |
| Agent CLI | ✅ `aegisx/agent_cli.py`: model output → gated tool call |
| Notebook (train) | ✅ no auto-push; manual export ZIP **with `knowledge/` folder + model card** for HF upload |
| Notebook (SFT own) | ✅ `notebooks/aegisx_sft_own_colab.ipynb` — stage 2: SFT of YOUR OWN from-scratch model (`--init-from`, no third-party base) |
| Notebook (finetune Qwen) | ✅ optional `aegisx_finetune_colab.ipynb` — QLoRA Qwen2.5-3B (only if user wants a non-pure-zero path) |
| Space app | ✅ RAG grounding: answers cite `knowledge/` sources (`hf/space_app.py`) |
| Known gap | Corpus should grow to 5–50 MB over time; add writeups & Q&A rows |

---

## 1. Data workstream — `@database` + `@researcher` + `@writer` ⭐ HIGHEST IMPACT

**Goal:** grow the corpus from ~15k to 500k–5M+ tokens of high-quality cysec text.

| # | Action | Effort | Impact |
|---|---|---|---|
| D1 | **Run `scripts/fetch_corpus.py`** (OWASP cheat sheets, ASVS, MITRE ATT&CK) — already wired into notebook §2b | 0 (done) | +500k–2M tokens |
| D2 | Add **CVE description dumps** (NVD public API → script) | M | teaches vuln patterns |
| D3 | Add **bug-bounty writeup corpus** (public HackerOne/Intigriti disclosures, MITRE cve.org) | M | teaches report style |
| D4 | Curate **Indonesian security content** (has some; expand: UU ITE, local labs, komunitas) | M | matches your usage language |
| D5 | **Q&A-style rows** for the chat format (`User: ... AegisX: ...`) so fine-tuning later starts from good structure | M | makes it assistant-like |
| D5b | ✅ **Indonesian chat corpus** — `data/raw/17_id_chat_bugbounty.txt` + `18_id_chat_defense.txt` (~25 KB, User:/AegisX: format) | done | fluency |
| D6 | **Dedup + clean** the corpus (script): strip URLs/banners, dedup near-identical paragraphs | S | quality > quantity |

**Rules (from `@writer`):** only public/properly-licensed text; no private data; no live target data; keep EN+ID balance.

---

## 2. Model & training workstream — `@ai-engineer` + `@perf`

**Goal:** train the biggest model that fits your hardware tier, efficiently.

| # | Action | Effort | Impact |
|---|---|---|---|
| M1 | ✅ **Right-size the model to data** — config raised to: `n_embd 512 / n_layer 8 / n_head 8 / block_size 256`, `max_steps 6000`, `warmup 300` (~30M params) | done | reasoning quality |
| M2 | **More steps** once data is big (max_steps 5k–20k is legitimate then — early stop keeps it honest) | S | fits big data |
| M3 | **Learning-rate warmup + cosine already in** ✅ | — | — |
| M4 | **Weight decay + grad clip already in** ✅ | — | — |
| M5 | ✅ **Periodic checkpoint** (`model_latest.pt` every eval) + best-weight restore — crash-safe long Colab runs | done | robustness |
| M6 | **Measure tokens/sec** on T4 vs. your laptop; tune `batch_size × grad_accum` so each step ≈ full GPU (report says 39k tok/s ✅ good) | S | speed |
| M7 | When data ≥5 MB: **train a second pass / longer schedule**, compare val loss | M | diminishing-returns check |

---

## 3. Quality & tests workstream — `@tddmaster` + `@qa` + `@refactor-expert`

**Goal:** every change lands verified. Dual-Gate: 0 static errors + behavioral tests.

| # | Action | Effort |
|---|---|---|
| Q1 | ✅ Add test: training on **tiny corpus with early stopping** completes and saves (guards the crash we fixed) | done |
| Q2 | ✅ Add test: `--tokenizer` resume path (train once, resume second run with same tokenizer) | done |
| Q3 | ✅ Add test: chat `generate()` one-shot mode returns non-empty string | done |
| Q4 | ✅ Add test: eval harness question set + coverage scoring | done |
| Q4 | Periodic **`@refactor-expert` pass**: kill dead code, keep modules small | ongoing |
| Q5 | After every feature: run `python3 -m pytest -q` (Gate 2) | always |

---

## 4. Knowledge layer (v0.3, next phase) — `@rag-vector-specialist` + `@nlp-rag-specialist`

**Goal:** ground answers in CVE/OWASP/writeup data so the small model stops hallucinating specifics.

| # | Action | Effort |
|---|---|---|
| R1 | Chunk the corpus (recursive, title+section metadata) | M |
| R2 | Embed locally (**BGE-M3**, no cloud) → Chroma/Qdrant | M |
| R3 | Hybrid search (BM25 + dense) + rerank → top-5 context | M |
| R4 | ✅ Wire retrieved context into the chat prompt — **done end-to-end**: `knowledge/` folder ships in the export ZIP; `hf/space_app.py` builds a `CorpusIndex` over it, grounds each answer, and shows `📚 Sumber: ...` under replies | done |

**Why it matters:** a 10–50M model cannot *know* CVE-2021-44228 details from weights alone; RAG gives it the document to quote. This is the single biggest capability boost after data.

---

## 5. Agent/tool layer — `@security` (already built; keep hardening)

**Goal:** AegisX can *do* recon/reporting, safely.

| # | Action | Effort |
|---|---|---|
| A1 | Keep **allowlist gate + audit log + no-sudo** (`aegisx/gate.py`) ✅ | — |
| A2 | Add an **e2e test** running a fake tool binary through the gate (subprocess path with no real tools installed) | S |
| A3 | Later: wire model `generate()` → agent tool selection (model picks tool, gate enforces) | L (v0.4) |

---

## 6. Release & deployment workstream — `@release`

**Goal:** you control when anything goes public. Manual by design.

| # | Action | When |
|---|---|---|
| Rel1 | Manual HF model upload (export ZIP from notebook §6 → drag-drop) | after a good run |
| Rel2 | Optional Gradio Space (follow `hf/README.md`) | when model is decent |
| Rel3 | Tag a **release** in GitHub (`v0.1.0`, `v0.2.0`…) matching HF uploads | each milestone |
| Rel4 | Keep `RFC.md` + this plan updated at each milestone | each milestone |

---

## 7. Suggested execution order (roadmap by `@planner`)

| Phase | Work | Exit criteria |
|---|---|---|
| **P1 (done)** | D1 corpus fetch ✅; run full Colab training on real data | val loss plateaus, chat output readable |
| **P2 (done)** | D2–D4 corpus expansion ✅ (CVE + OWASP + ATT&CK); Q1–Q3 tests ✅ | model live on HF (manual upload when ready) |
| **P3 (done)** | R1–R4 RAG knowledge layer ✅ (zero-dep keyword index, df-cache) | grounded answers with source ✅ |
| **P3b (done)** | RAG wired into Space app ✅ (knowledge/ folder ships with export) | Space answers cite sources ✅ |
| **P4 (done)** | A3 agent↔model wiring ✅ (`agent_cli.py`, gated) | agent demo on authorized lab target |
| **P5 (done)** | **Pure-zero SFT** ✅ — stage-2 notebook (`--init-from` your own model + 925-row SFT text); QLoRA-Qwen kept only as optional non-zero alternative | model answers questions, still 100% own weights |

**Next up:** run the full Colab training on the new 2.3 MB corpus with the bigger
config (M1: `n_embd 512 / n_layer 8`), then run §5b eval and upload when the
numbers look right. Expected T4 time: tokenizer ~1–2 min, then training until
early stop (watch for `🛑`).

**Golden rule:** data → model → measure → repeat. Every iteration ends with a
val-loss number and a chat sample, so we can see improvement, not guess it.

---

*Directives honored from every sub-agent: zero placeholder stubs, strict typing,
dual-gate verification (static + behavioral), self-heal before finishing.*