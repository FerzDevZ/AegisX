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
| Pipeline | ✅ **Satu tombol ①→②→③** (`scripts/pipeline_full.py` + `notebooks/aegisx_pipeline_colab.ipynb`): pre-train → SFT → DPO, auto-resume, auto-skip tahap selesai, auto-arsip checkpoint arsitektur lama, export ZIP final. Teruji end-to-end di CPU + Colab |
| Tests | ✅ **84/84 passing** (tokenizer, model, training, gate, RAG v2, agent CLI ReAct, DPO, eval, clean, expand) |
| Corpus | ✅ **~46 MB / 212 file** — OWASP (ASVS + CheatSheetSeries ±120 sheet + WSTG + MASTG + API-Security + Top10), MITRE ATT&CK (enterprise + ICS + mobile), PayloadsAllTheThings, **HackTricks (CC BY-NC, atribusi ada)**, CVE (NVD bulk ~38k), custom EN+ID, 21 file chat Indonesia `User:/AegisX:`, Wikipedia ID keamanan siber (fetch via Special:Export, resume-safe) |
| Corpus cleaner | ✅ `scripts/clean_corpus.py` dedup + strip noise (`--apply`, .bak backup) + **sanitasi pola secret** (placeholder `AKIA-REDACTED`, contoh AWS key/GitHub PAT palsu di dokumen publik) |
| Language audit | ✅ `scripts/audit_language.py` — korpus mentah: ID 2.8% / EN 97.2% (gap utama, lihat D4b) |
| Architecture | ✅ **vocab 8192 / block 768 / n_layer 8 / n_head 8 / n_embd 512** (±30M params, tied embedding) — context 3× lebih panjang, BPE dwibahasa lebih efisien |
| Tokenizer | ✅ byte-level BPE, dilatih dari korpus sendiri, disimpan ke Drive (`tokenizer.json`) — hanya sekali per vocab |
| Training | ✅ AMP (T4 ±1.5–2×), early stop + restore bobot terbaik, history CSV, `model_latest.pt` tiap eval (crash-safe), warmup + cosine, weight decay + grad clip |
| RAG | ✅ **RAG v2** (`aegisx/rag.py`, zero-dep): chunking section-aware + breadcrumb `§ Bab > Sub-bab`, **BM25 asli** (k1=1.5, b=0.75), rerank heading/diversity, ambang skor relatif. API kompatibel ke belakang |
| Eval | ✅ `aegisx/eval.py`: 20 fixed Q (10 EN + 10 ID), keyword-coverage scoring, `--history` CSV antar-run |
| Agent CLI | ✅ `aegisx/agent_cli.py`: **ReAct loop** (rencana→tool→refleksi), max 5 tool-round (circuit breaker), `--confirm` (human-in-the-loop), `--allowlist` + audit log, **sesi SQLite** (`--session`/`--resume`) |
| Dataset instruksi | ✅ **1.371 baris** (1.101 ID = 80.3% / 270 EN) — `data/finetune/instructions.jsonl` |
| Preference (DPO) | ✅ **1.386 pasangan chosen/rejected** (`data/finetune/preferences.jsonl`), mayoritas ID, termasuk penolakan sopan utk 15 tipe permintaan berbahaya |
| Alignment (tahap-3) | ✅ **DPO-lite** (`aegisx/dpo.py`): reference model beku, beta 0.05, loss panjang-dinormalisasi, resume + history; `scripts/build_preferences.py`; notebook `aegisx_align_own_colab.ipynb` |
| Notebook (train) | ✅ `notebooks/aegisx_train_colab.ipynb` — no auto-push; manual export ZIP **dengan `knowledge/` + model card** |
| Notebook (SFT own) | ✅ `notebooks/aegisx_sft_own_colab.ipynb` — SFT model dari-scratch sendiri (`--init-from`, tanpa base pihak ketiga) |
| Notebook (pipeline) | ✅ `notebooks/aegisx_pipeline_colab.ipynb` — Run all = fetch (opsional) → ①→②→③ → uji chat → export `aegisx-mini-final.zip` |
| Notebook (alternatif) | optional `aegisx_finetune_colab.ipynb` (QLoRA Qwen2.5-3B — hanya bila mau jalur non-pure-zero) + varian Kaggle (`aegisx_train_kaggle.ipynb`, `aegisx_sft_kaggle.ipynb`) |
| Space app | ✅ int8 dynamic quantization (CPU, 2–3× lebih cepat, memori ~4× lebih kecil) + **streaming token** di CPU; ZeroGPU = fp32 one-shot; RAG v2 grounding `knowledge/` + `📚 Sumber:` |
| Known gap | Porsi bahasa Indonesia korpus mentah masih **2.8%** — perbanyak Wikipedia ID & artikel ID ke 15–20% (D4b). Eval masih 20 soal → target 100+ (Q6). Guardrails berlapis + tool-call format andal belum (v1.0) |

---

## 1. Data workstream — `@database` + `@researcher` + `@writer` ⭐ HIGHEST IMPACT

**Goal:** grow the corpus toward 100–200 MB high-quality cysec text, and lift the
Indonesian share from 2.8% to 15–20% of bytes.

| # | Action | Effort | Impact |
|---|---|---|---|
| D1 | ✅ `scripts/fetch_corpus.py` (OWASP, ASVS, MITRE ATT&CK, WSTG, PayloadsAllTheThings, HackTricks, MASTG, API-Security, Top10, NVD CVE, Wikipedia ID) — ter-wire ke notebook §2b | done | +korpora besar, lisensi sah |
| D2 | ✅ CVE description dumps (NVD API bulk) | done | teaches vuln patterns |
| D3 | ✅ Bug-bounty writeup corpus (HackTricks, MASTG…) | done | teaches report style |
| D4 | ✅ Indonesian security content (chat ID 21 file + Wikipedia ID) | done | matches usage language |
| D4b | 🔶 **Perbesar korpus ID ke 15–20% byte** — Wikipedia ID kategori dalam (jalankan fetch berulang), advisori BSSN/ID-CERT, artikel teknologi ID, UU ITE | M | **gap utama sekarang** |
| D5 | ✅ Q&A-style rows (`User:/AegisX:`) — 1.371 instruksi (80.3% ID) | done | makes it assistant-like |
| D6 | ✅ Dedup + clean corpus (`clean_corpus.py`) + sanitasi secret | done | quality > quantity |
| D7 | 🔶 **Prefer narasi panjang** (PortSwigger research, HackerOne disclosed) di atas daftar CVE — narasi mengajarkan rantai penalaran | M | reasoning |
| D8 | 🔶 **Multi-turn conversation** di data SFT (follow-up, klarifikasi) | M | real chat is multi-turn |

**Rules (from `@writer`):** only public/properly-licensed text (CC BY-SA /
CC BY-NC dengan atribusi); no private data; no live target data; keep EN+ID balance.

---

## 2. Model & training workstream — `@ai-engineer` + `@perf`

**Goal:** train the biggest model that fits your hardware tier, efficiently.

| # | Action | Effort | Impact |
|---|---|---|---|
| M1 | ✅ Arsitektur dinaikkan: **vocab 8192 / block 768 / n_embd 512 / n_layer 8 / n_head 8** (±30M params) | done | konteks 3×, BPE dwibahasa |
| M2 | ✅ Auto-arsip checkpoint arsitektur lama (bandingkan struktural block/layer/head/embd — bukan vocab, karena BPE byte-fallback bikin vocab aktual ≠ target) | done | retrain bersih tanpa crash |
| M3 | ✅ LR warmup + cosine · weight decay + grad clip · AMP · early stop + restore best | done | stabilitas |
| M4 | ✅ Periodic checkpoint `model_latest.pt` (crash-safe) + `history.csv` | done | robustness |
| M5 | ✅ Resume dengan LR lebih kecil (anti-catastrophic-forgetting) saat lanjut dari checkpoint | done | continual pre-train aman |
| M6 | Tokenizer hanya dilatih sekali per vocab → disimpan; run berikutnya load dalam detik | done | hemat GPU |
| M7 | Naikkan `max_steps` seiring data bertambah (5k–20k sah setelah data besar; early stop menjaga kejujuran) | S | fits big data |

Hasil terukur (run terbaik, arsitektur lama 4096/256): val loss **2.4387**
(perplexity ≈ 11.5) dengan jarak train/val sehat. Run pada arsitektur baru
(8192/768, korpus 46 MB) adalah tolok ukur berikutnya.

---

## 3. Quality & tests workstream — `@tddmaster` + `@qa` + `@refactor-expert`

**Goal:** every change lands verified. Dual-Gate: 0 static errors + behavioral tests.

| # | Action | Effort |
|---|---|---|
| Q1 | ✅ Training tiny corpus + early stopping selesai & menyimpan (guard crash yang pernah terjadi) | done |
| Q2 | ✅ Resume tokenizer (train sekali, resume dengan tokenizer sama) | done |
| Q3 | ✅ Chat `generate()` one-shot mengembalikan string non-kosong | done |
| Q4 | ✅ Eval harness + coverage scoring | done |
| Q5 | ✅ Tes DPO: loss memilih chosen, normalisasi panjang, log-prob cocok manual, builder pasangan | done |
| Q6 | 🔶 **Eval suite 100+ soal** (50 ID + 50 EN, multi-turn) + laporan HTML per run; jadikan gate sebelum upload | M |
| Q7 | Periodic `@refactor-expert` pass: kill dead code, keep modules small | ongoing |
| Q8 | Setelah tiap fitur: `python3 -m pytest -q` (Gate 2) | always |

---

## 4. Knowledge layer — `@rag-vector-specialist` + `@nlp-rag-specialist`

**Goal:** ground answers in CVE/OWASP/writeup data so the small model stops
hallucinating specifics.

| # | Action | Effort |
|---|---|---|
| R1 | ✅ Chunking section-aware + breadcrumb 2 level (`§ Bab > Sub-bab`) | done |
| R2 | ✅ **BM25 asli** (k1=1.5, b=0.75, IDF sesungguhnya) menggantikan token-overlap | done |
| R3 | ✅ Rerank: bonus kata kunci di area heading + diversity per sumber; ambang skor relatif (<25% skor terbaik dibuang) | done |
| R4 | ✅ `knowledge/` folder ikut dalam export ZIP; `hf/space_app.py` membangun index + `📚 Sumber:` | done |
| R5 | 🔶 RAG v3 (nanti): embed lokal (BGE-M3) + hybrid BM25+dense; perlu resource lebih | L (v2) |

**Why it matters:** a 10–50M model cannot *know* CVE-2021-44228 details from
weights alone; RAG gives it the document to quote. This is the single biggest
capability boost after data.

---

## 5. Agent/tool layer — `@security` + `@ai-safety-guardrails-redteam`

**Goal:** AegisX can *do* recon/reporting, safely, with human-in-the-loop.

| # | Action | Effort |
|---|---|---|
| A1 | ✅ Allowlist gate + audit log + no-sudo (`aegisx/gate.py`) | — |
| A2 | ✅ **ReAct loop** di `agent_cli.py`: rencana→tool→refleksi, max 5 round (circuit breaker) | done |
| A3 | ✅ `--confirm` (minta y/N sebelum tool) + sesi SQLite (`--session`/`--resume`) | done |
| A4 | 🔶 Tool-call format andal yang bisa di-parse model secara konsisten (`[RUN <tool> <args>]` di data SFT) | M |
| A5 | 🔶 Guardrails berlapis: input guard (prompt injection) → model → output guard (red-team) | M (v1.0) |

---

## 6. Alignment workstream (tahap-3) — `@llm-finetuning-post-training`

**Goal:** after SFT, make AegisX *behave* — follow the user, refuse off-scope or
harmful asks, answer in Indonesian.

| # | Action | Effort |
|---|---|---|
| P1 | ✅ DPO-lite trainer (`dpo.py`): reference model beku, beta 0.05, loss panjang-dinormalisasi, resume + history | done |
| P2 | ✅ 1.386 pasangan preferensi ID (`preferences.py` + `build_preferences.py`) — termasuk penolakan sopan utk 15 tipe permintaan berbahaya | done |
| P3 | ✅ Ter-orchestrasi dalam `pipeline_full.py` (tahap 3) + notebook mandiri `aegisx_align_own_colab.ipynb` | done |
| P4 | 🔶 Data preferensi lebih banyak (2–5K) + variasi penolakan agar model tidak overfit ke satu pola kalimat | M |

---

## 7. Release & deployment workstream — `@release`

**Goal:** you control when anything goes public. Manual by design.

| # | Action | When |
|---|---|---|
| Rel1 | Manual HF upload (export ZIP dari notebook → drag-drop ke Space) | after a good run |
| Rel2 | Space app: int8 + streaming + RAG v2 (`hf/space_app.py`) — sudah live | now |
| Rel3 | Tag a **release** in GitHub (`v0.1.0`, `v0.2.0`…) matching HF uploads | each milestone |
| Rel4 | Keep `RFC.md` + this plan updated at each milestone | each milestone |

---

## 8. Suggested execution order (roadmap by `@planner`)

| Phase | Work | Exit criteria |
|---|---|---|
| **P1 (done)** | Corpus fetch + Colab training pada data nyata | val loss plateau, chat readable |
| **P2 (done)** | Ekspansi korpus (OWASP, ATT&CK, CVE, HackTricks, MASTG, Top10); tes Q1–Q3 | model live di HF (manual) |
| **P3 (done)** | RAG v1 → **v2** (BM25 + section-aware) + `knowledge/` di export | grounded answers + source |
| **P4 (done)** | Agent↔model: ReAct loop + `--confirm` + sesi SQLite | agent demo di target lab |
| **P5 (done)** | Pure-zero SFT (1.371 instruksi) — tanpa base pihak ketiga | model menjawab pertanyaan |
| **P6 (done)** | Tahap-3 DPO (1.386 preferensi ID) + pipeline 1-tombol ①→②→③ | model menolak permintaan berbahaya |
| **P7 (done)** | Korpus 46 MB + arsitektur vocab 8192/block 768 + int8/streaming Space | run arsitektur baru di T4 |
| **P8 (next)** | Retrain bersih arsitektur baru di Colab (tokenizer 8192 sekali ±30–90 mnt → train ±1–1,5 jam → SFT → DPO) → export → upload | val loss < baseline & chat ID fasih |
| **P9 (next)** | Korpus ID 2.8% → 15–20% (D4b) + narasi panjang (D7) + multi-turn (D8) | audit_language menunjukkan target |
| **P10 (next)** | Eval suite 100 soal + laporan HTML (Q6); jadikan gate upload | tiap retrain terukur |

**Golden rule:** data → model → measure → repeat. Every iteration ends with a
val-loss number and a chat sample, so we can see improvement, not guess it.

---

*Directives honored from every sub-agent: zero placeholder stubs, strict typing,
dual-gate verification (static + behavioral), self-heal before finishing.*
