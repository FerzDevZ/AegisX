# ⚡ AegisX-Mini

**Model AI keamanan siber yang dilatih *dari nol* (pure zero)** — ringan, gratis
dilatih, dan dirancang untuk bertumbuh menjadi asisten bug-bounty & keamanan
siber dwibahasa (Indonesia + Inggris).

> Dibangun sepenuhnya dengan **weight sendiri** — bukan fine-tune dari model
> pihak ketiga. Tidak ada Qwen/Llama/Mistral di dalamnya.

| | |
|---|---|
| **Model** | GPT-style decoder-only transformer, ±30M parameter |
| **Pipeline** | 3 tahap otomatis: **① pre-train → ② SFT → ③ DPO alignment** |
| **Tokenizer** | Byte-level BPE (vocab 8.192, dilatih dari korpus sendiri) |
| **Konteks** | 768 token (3× lebih panjang dari v1) |
| **Training** | Google Colab gratis (T4) / Kaggle — laptop tidak pernah melatih |
| **Hosting** | Hugging Face Space (Gradio, int8 + streaming, RAG grounding) |
| **RAG** | Retrieval tanpa dependensi eksternal: BM25 + chunking per-bagian |
| **Agent layer** | ReAct loop dengan *authorization gate* + audit log + konfirmasi manusia |
| **Tes** | 84/84 hijau |

> 📐 Arsitektur & keputusan desain: [RFC.md](RFC.md) · 📋 Rencana pengembangan
> hidup: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)

---

## 🚀 Cara tercepat: SATU TOMBOL di Colab

Seluruh pipeline (pre-train → SFT → DPO → uji → export ZIP) sudah diorkestrasi
dalam satu notebook. Tidak perlu menjalankan tahap satu-satu.

1. Buka **`notebooks/aegisx_pipeline_colab.ipynb`** di
   [Google Colab](https://colab.research.google.com/) (File → Upload notebook,
   atau buka langsung dari GitHub → "Open in Colab").
2. **Runtime → Run all** — selesai.
3. Hasil akhir ada di Google Drive:
   `MyDrive/aegisx/export/aegisx-mini-final.zip` → upload manual ke Hugging Face.

Script orkestratornya (`scripts/pipeline_full.py`) **cerdas** soal resume:

| Situasi | Yang terjadi otomatis |
|---|---|
| Tahap sudah selesai | Di-*skip* (tidak buang waktu GPU) |
| Sesi Colab putus di tengah training | Lanjut dari `model_latest.pt` |
| Ada checkpoint arsitektur lama di Drive | Di-*arsip* dulu → training bersih, tidak crash |
| Cuma mau SFT ulang setelah ganti data | Atur `STAGES = "2"` di notebook |
| Mau paksa ulang semua | Tambahkan `--force` |

Perkiraan waktu di T4 gratis (sekali jalan): tokenizer ±30–90 menit (hanya
sekali seumur arsitektur, langsung tersimpan ke Drive) → training 1500 step
±1–1,5 jam → SFT ±15–20 menit → DPO ±10 menit.

---

## 🧠 Pipeline 3 tahap

| Tahap | Output | Isi |
|---|---|---|
| **① Pre-train** | `checkpoints/aegisx-mini/` | Belajar bahasa dari korpus mentah (46 MB, dwibahasa). Hasilnya model yang fasih *meneruskan* teks keamanan siber |
| **② SFT** (supervised fine-tune) | `checkpoints/aegisx-sft/` | DIAJARKAN format tanya-jawab `User: … AegisX: …` dari 1.371 instruksi (80% Indonesia) → mulai *menjawab*, bukan sekadar meneruskan teks |
| **③ DPO alignment** | `checkpoints/aegisx-align/` | DIAJARKAN preferensi dari 1.386 pasangan *jawaban bagus vs buruk* → mengikuti pengguna, **menolak permintaan di luar cakupan/berbahaya** dengan sopan dalam bahasa Indonesia |

Tiap tahap menyimpan `model.pt` + `tokenizer.json` + `config.json` +
`history.csv` (riwayat val loss) + `model_latest.pt` (checkpoint berkala).

---

## 💻 Quickstart lokal (CPU, tes cepat)

```bash
# 1. Install (cukup torch + pytest)
pip install -r requirements.txt

# 2. Train model mini dari korpus seed (config kecil biar cepat di CPU)
python -m aegisx.train --data data/raw --out checkpoints/aegisx-mini \
    --vocab-size 2048 --block-size 128 --n-layer 2 --n-head 2 --n-embd 128 \
    --max-steps 50 --eval-every 25 --save-every 25

# 3. Ngobrol dengan model
python -m aegisx.chat --model checkpoints/aegisx-mini/model.pt \
    --tokenizer checkpoints/aegisx-mini/tokenizer.json

# 4. Evaluasi pada 20 soal tetap
python -m aegisx.eval --model checkpoints/aegisx-mini/model.pt \
    --tokenizer checkpoints/aegisx-mini/tokenizer.json
```

> Bagian paling lambat di laptop adalah **membangun tokenizer** dari `data/raw/`
> (beberapa menit untuk vocab 2048). Untuk smoke test tercepat, arahkan
> `--data` ke folder kecil berisi 2–3 file `.txt` (`mkdir data/demo && cp
> data/raw/*.txt data/demo/` secukupnya) — atau pakai tokenizer yang sudah ada
> lewat `--tokenizer path/tokenizer.json`. Training sungguhan tetap di Colab/Kaggle.

Arsitektur diatur lewat argumen `--vocab-size --block-size --n-layer --n-head
--n-embd`. Contoh config yang dipakai untuk korpus 46 MB:

```bash
python -m aegisx.train --data data/raw --out checkpoints/aegisx-mini \
    --vocab-size 8192 --block-size 768 --n-layer 8 --n-head 8 --n-embd 512 \
    --batch-size 16 --grad-accum 4 --max-steps 1500 --lr 3e-4 \
    --warmup-steps 400 --eval-every 300 --save-every 100 \
    --early-stop-patience 5 --device cuda
```

> Fitur otomatis: **AMP** (mixed precision, T4 ±1,5–2× lebih cepat, nonaktifkan
> dengan `--no-amp`) · **early stop** (berhenti saat val loss datar, memulihkan
> bobot terbaik) · **resume** (`--init-from` + `model_latest.pt`) · **history
> CSV** tiap eval.

---

## 📁 Struktur proyek

```
aegisx/
  tokenizer.py    # byte-level BPE: train / encode / decode / save / load
  model.py        # GPT decoder-only transformer (config, generate, save/load)
  train.py        # pre-training: dataset, tokenizer, checkpoint, early-stop, AMP
  chat.py         # chat interaktif + one-shot generation
  eval.py         # evaluasi 20 soal tetap + skor coverage + riwayat CSV
  rag.py          # RAG v2: chunking per-bagian + BM25 + rerank (tanpa dependensi)
  dpo.py          # trainer tahap-3: DPO-lite (reference model beku, beta 0.05)
  preferences.py  # template pasangan preferensi (chosen/rejected) bahasa Indonesia
  gate.py         # authorization gate: allowlist + audit log + eksekusi aman
  agent.py        # tool agent (recon, web, code search, laporan) — ter-gate
  agent_cli.py    # agentic chat: ReAct loop + --confirm + sesi SQLite
data/
  raw/            # korpus mentah (.txt) — 212 file, ±46 MB. Tambahkan sendiri
  finetune/       # instructions.jsonl (1.371) · preferences.jsonl (1.386) · sft_chat.txt
scripts/
  pipeline_full.py      # orkestrator ①→②→③ satu perintah
  fetch_corpus.py       # unduh korpus publik (OWASP, MITRE, CVE, HackTricks, Wikipedia ID…)
  clean_corpus.py       # dedup + bersihkan noise (--apply untuk menulis)
  build_instructions.py # file chat → instructions.jsonl
  expand_instructions.py# parafrase untuk memperbesar dataset instruksi
  build_sft_text.py     # instructions.jsonl → teks SFT User:/AegisX:
  build_preferences.py  # bangun pasangan preferensi DPO
  audit_language.py     # ukur rasio bahasa EN vs ID
targets/
  authorized.txt   # daftar target yang diizinkan (edit!)
notebooks/         # notebook Colab & Kaggle (train, SFT, DPO, pipeline 1-tombol)
hf/                # template Space + model card + push script
tests/             # 84 tes pytest
```

---

## 📚 Tentang data

### Korpus mentah (`data/raw`) — ±46 MB / 212 file
- **OWASP**: Cheat Sheet Series (±120 sheet), ASVS, WSTG, API-Security, Top10, MASTG
- **MITRE ATT&CK** (enterprise + ICS + mobile), PayloadsAllTheThings
- **HackTricks** (lisensi CC BY-NC 4.0, atribusi ada)
- **CVE** (NVD bulk, puluhan ribu deskripsi)
- **Wikipedia ID** keamanan siber + file chat Indonesia (21 file format `User:/AegisX:`)

Semua sumber publik & berlisensi sah (CC BY-SA / CC BY-NC dengan atribusi).
Jangan pernah menambah data privat atau data target live.

```bash
# Menambah korpus publik baru
python3 scripts/fetch_corpus.py --target-dir data/raw
# CVE saja (skip NVD bulk besar): --cve-pages 0
# Bersihkan setelah menambah:  python3 scripts/clean_corpus.py --apply
# Ukur rasio bahasa:           python3 scripts/audit_language.py
```

> **Gap yang diketahui:** bahasa Indonesia baru ±2,8% dari byte korpus mentah.
> Target: 15–20%. Instruksi SFT sudah kebalikannya — 80% Indonesia. Tambahan
> korpus ID (advisori BSSN/ID-CERT, artikel ID, Wikipedia ID) adalah prioritas
> tertinggi siklus berikutnya.

### Dataset instruksi (`data/finetune`) — 1.371 baris
- **80,3% Indonesia / 19,7% Inggris**
- Topik: pentest methodology, OWASP (semua kategori), OSINT, forensik,
  malware, wireless, jaringan, UU ITE, defense, laporan bug-bounty
- Format `User: … AegisX: …` — dibangun ulang dengan:
  `python3 scripts/build_instructions.py --force`

### Pasangan preferensi DPO — 1.386 pasangan
`chosen` = jawaban bagus untuk pertanyaan sah; `rejected` = jawaban buruk.
Termasuk **penolakan sopan bahasa Indonesia** untuk 15 tipe permintaan
berbahaya (meretas akun, DDOS, mencuri kartu, dsb). Bangun ulang:
`python3 scripts/build_preferences.py`

---

## 🔎 RAG grounding (jawaban bersumber)

Model kecil tidak bisa *menghafal* detail semua CVE dari bobot saja — RAG
memberi model dokumen untuk dikutip. `aegisx/rag.py` tanpa dependensi eksternal:
chunking per-bagian (breadcrumb `§ Bab > Sub-bab`), BM25, rerank + ambang skor.

```python
from aegisx.rag import CorpusIndex

idx = CorpusIndex()
idx.add_dir('data/raw')
for chunk in idx.retrieve('sql injection prevention', top_k=3):
    print(f'[{chunk.source}] {chunk.text[:200]}')
```

RAG otomatis aktif di **Space Hugging Face**: setiap jawaban yang grounded
menampilkan baris `📚 Sumber: …` di bawahnya.

---

## 🛡️ Agent mode (bug-bounty, terkendali)

`aegisx/agent_cli.py` menjalankan **ReAct loop**: model menyusun rencana →
mengeksekusi tool → merefleksikan hasil → memutuskan lanjut atau menjawab.
Semua aksi ofensif melewati `AuthorizationGate`:

1. **Target allowlist** — edit `targets/authorized.txt`; di luar daftar = **ditolak**
2. **Audit log** — setiap aksi dicatat berstempel waktu di `logs/audit.log`
3. **Eksekusi aman** — `sudo` & metakarakter shell diblokir; perintah jalan sebagai `argv`
4. **Konfirmasi manusia** — `--confirm` meminta `y/N` sebelum tool berjalan
5. **Circuit breaker** — maks. 5 round tool per giliran (anti-loop)

```bash
python -m aegisx.agent_cli \
    --model checkpoints/aegisx-align/model.pt \
    --tokenizer checkpoints/aegisx-align/tokenizer.json \
    --allowlist targets/authorized.txt --confirm --resume
```

Pemakaian dari Python:

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

> ⚠️ Hanya jalankan terhadap host yang **kamu miliki atau punya izin tertulis**.
> Bug bounty = aset yang masuk *scope* program saja.

---

## 🧪 Verifikasi

```bash
python3 -m pytest -q        # 84 passed
```

Cakupan tes: tokenizer round-trip, forward/training/generasi model, dataset
split, early-stop end-to-end, resume tokenizer, one-shot chat, authorization
gate (allowlist, audit log, blokir sudo/injeksi shell), RAG retrieval & BM25,
agent-CLI ReAct, DPO (loss memilih chosen, normalisasi panjang, log-prob vs
manual), clean-corpus, expand-instructions, dan eval scoring.

---

## ☁️ Deploy ke Hugging Face (manual — kamu yang pegang kendali)

Pipeline **tidak pernah auto-push** — upload selalu manual supaya kamu bisa
memeriksa hasil dulu. Langkah lengkap: [hf/README.md](hf/README.md).

Intinya:

1. Setelah pipeline selesai, unduh `aegisx-mini-final.zip` dari Drive
2. Di Space HF (`FerzDevZ/aegisx-mini-space`): ganti `model.pt`,
   `tokenizer.json`, `config.json`, unggah folder `knowledge/` (+ `README.md`
   model card)
3. Space rebuild otomatis → buka URL publik → tes chat

---

## 🗺️ Roadmap

- [x] v0.1 — tokenizer, model, training CPU, chat, authorization gate
- [x] v0.2 — Colab T4 (AMP, resume, early-stop) + export manual → Space Gradio live
- [x] v0.3 — RAG v1 (keyword) → **RAG v2** (BM25 + chunking per-bagian) di Space
- [x] v0.4 — agent ↔ model (ReAct loop, gate, `--confirm`, sesi SQLite)
- [x] v0.5 — **pure-zero SFT** (format `User:/AegisX:`, 1.371 instruksi)
- [x] v0.6 — **tahap-3 DPO alignment** (1.386 preferensi ID) + pipeline 1-tombol
- [x] v0.7 — korpus 46 MB, arsitektur vocab 8.192 / block 768, int8 + streaming di Space
- [ ] v0.8 — porsi bahasa Indonesia korpus 2,8% → 15–20%
- [ ] v0.9 — eval suite 100+ soal (50 ID + 50 EN) + laporan HTML per run
- [ ] v1.0 — guardrails berlapis & tool-call format yang andal untuk agent

**Aturan emas:** data → model → ukur → ulangi. Setiap iterasi berakhir dengan
angka val loss + sampel chat, supaya kemajuan terlihat, bukan diterka.

---

## 📄 Dokumen lain

| Dokumen | Isi |
|---|---|
| [RFC.md](RFC.md) | Arsitektur, desain, trade-off, STRIDE security matrix |
| [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) | Rencana pengembangan hidup + status tiap workstream |
| [hf/README.md](hf/README.md) | Panduan deploy manual ke Hugging Face |
| [hf/MODEL_CARD.md](hf/MODEL_CARD.md) | Model card (ikut ter-export ke ZIP) |
