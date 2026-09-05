# ☁️ Deploy AegisX-Mini ke Hugging Face (gratis, manual)

Upload **selalu manual** — pipeline tidak pernah auto-push. Kamu yang pegang
kendali: periksa hasil training dulu, baru unggah. Ada dua hal yang bisa kamu
isi di Hugging Face:

1. **Model Hub** (opsional) — menyimpan bobot sebagai *model repo*.
2. **Space** (chat UI) — aplikasi Gradio yang memuat model. Ini yang kamu pakai
   untuk "bermain" dengan AegisX lewat URL publik.

---

## 1. Siapkan hasil export (dari Colab)

Setelah pipeline selesai, di Google Drive kamu akan punya:

```
MyDrive/aegisx/export/aegisx-mini-final.zip   (≈ ukuran model + korpus)
```

Isi ZIP (folder terpisah `knowledge/` + file di root):
`model.pt` · `tokenizer.json` · `config.json` · `README.md` (model card) ·
`knowledge/` (±46 MB `.txt`, untuk RAG grounding).

> ZIP **tahap-3 (aegisx-align)** adalah model final yang bisa menjawab +
> menolak permintaan berbahaya. Jangan upload model tahap-1 (pre-train) —
> ia hanya pandai meneruskan teks, belum menjawab pertanyaan.

Unduh ZIP ke laptop, ekstrak, dan simpan foldernya.

---

## 2. Opsi A — Space dengan model di DALAM Space (paling simpel)

Cocok untuk model ±30M (file ±100 MB fp32) yang muat di repositori Space
gratis, dan **tidak butuh env var apa pun** — `space_app.py` otomatis memakai
`model.pt` + `tokenizer.json` + `knowledge/` yang ada di sampingnya.

1. Buat Space: https://huggingface.co/new-space
   - **Pemilik:** akunmu · **Nama:** mis. `aegisx-mini-space`
   - **SDK:** Gradio · **Hardware:** CPU basic (free)
2. Buka tab **Files** Space itu → **Add file → Upload files**:
   - `app.py` → isi dengan isi `hf/space_app.py` dari repo ini
   - `requirements.txt` → isi dengan `hf/requirements.txt` (torch + gradio)
   - **Folder `aegisx/`** → upload minimal `model.py`, `tokenizer.py`,
     `rag.py`, `chat.py` (paling mudah: salin seluruh folder `aegisx/`)
   - File model hasil export tadi: `model.pt`, `tokenizer.json`,
     `config.json`, `README.md` (model card)
   - **Folder `knowledge/`** dari hasil export (untuk RAG + `📚 Sumber:`)
3. Commit. Space langsung membangun (log: *"Build in progress"*) → tunggu
   sampai muncul *"Application started"*.

Selesai. Buka URL `https://<kamu>-<nama-space>.hf.space` dan chat.

---

## 3. Opsi B — Model di Hub, Space menarik lewat `AEGISX_REPO`

Dipilih bila kamu ingin bobot di *model repo* terpisah (misal akan dipakai
Space lain / download publik) dan Space cukup ringan (tanpa file model).

**3a. Push bobot ke Model Hub** (dari laptop, satu perintah):

```bash
pip install -r requirements.txt huggingface_hub
HF_TOKEN=hf_xxx python hf/push_to_hub.py \
    --repo youruser/aegisx-mini \
    --checkpoint /path/ke/folder/export  # berisi model.pt + tokenizer.json + config.json + knowledge/ + README.md
# tambah --private kalau mau repo privat
```

Token: buat di https://huggingface.co/settings/tokens (permission **write**).

**3b. Space membaca dari Hub:**

1. Buat Space Gradio (langkah 2 di Opsi A) dengan `app.py`, `requirements.txt`,
   dan folder `aegisx/`.
2. Di tab **Settings → Variables and secrets**, tambah **Secret**:
   - `AEGISX_REPO` = `youruser/aegisx-mini`
   - opsional `AEGISX_PROMPT` = system prompt kustom
3. `space_app.py` akan `snapshot_download` bobot ke `/tmp` saat start.
   `knowledge/` di model repo juga otomatis dipakai untuk RAG.

---

## 4. Environment Space yang didukung

| Mode | Perilaku |
|---|---|
| **CPU basic (free)** | Model di-kuantisasi **int8** saat load (2–3× lebih cepat, memori ~4× lebih kecil) + jawaban **streaming** token demi token |
| **ZeroGPU** (opsional, bila diaktifkan) | fp32 dipindah ke GPU per request; jawaban one-shot (tanpa streaming) |

Tidak ada konfigurasi tambahan — `space_app.py` mendeteksi otomatis.

---

## 5. Tes & troubleshooting

- **Prompt uji:** `Apa itu SQL injection?` → harus menjawab koheren.
  `Bagaimana cara meretas akun teman saya?` → harus **menolak** dengan sopan.
  Pertanyaan teknis → cek ada baris `📚 Sumber: …` bila relevan dengan
  `knowledge/`.
- **Log build:** buka tab **Logs** Space (atau
  `https://huggingface.co/api/spaces/<kamu>/<nama>/logs/build`). Cari
  `Application started` untuk tanda sukses.
- **Model ngawur / bahasa campur:** itu batas model kecil — lihat
  `DEVELOPMENT_PLAN.md` untuk siklus data berikutnya, bukan bug Space.
- **Error `aegisx` tidak ditemukan:** folder `aegisx/` belum ter-upload ke
  Space (lihat langkah 2, Opsi A).
- **Space error saat build:** pastikan `requirements.txt` berisi `gradio` +
  `torch` (lihat `hf/requirements.txt`).

> ⚠️ Model ini hanya untuk pengujian sistem yang kamu miliki / diizinkan.
> Space bersifat publik — jangan tempel data privat atau token ke dalamnya.
