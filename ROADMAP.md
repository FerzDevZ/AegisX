# 🗺️ ROADMAP — AegisX Menuju Powerful & Nyambung

> Disusun 2026-09-06, berdasarkan diagnosis nyata hari ini: model hasil export
> terakhir (arsitektur 8192/768) ternyata **nyaris acak** — NLL 9,9–16,8 pada
> teks Indonesia (lebih buruk dari menebak acak ln(8192)=9,01) dan malah
> menghasilkan karakter Jepang. VPS sehat (HTTP 200, systemd aktif) tapi yang
> dilayani adalah model rusak. Dokumen ini rencana perbaikan + penguatan
> berjenjang.

**Prinsip kerja:** data → model → **ukur** → ulangi. Tidak ada tahap yang
lanjut sebelum tahap sebelumnya lolos quality gate. Setiap fase punya
**kriteria keluar yang terukur** — bukan perasaan.

---

## 📊 Posisi saat ini (faktor yang sudah dicek)

| Komponen | Status | Bukti |
|---|---|---|
| VPS AWS (2 vCPU / 8 GB) | ✅ SEHAT | HTTP 200 di `:7860`, systemd `aegisx.service` aktif, auto-restart |
| Deploy pipeline (rsync → venv → systemd) | ✅ PROVEN | 141 MB terkirim, torch 2.14 CPU + gradio 6.26 |
| Tokenizer 8192 | ✅ SEHAT | round-trip 100% benar, vocab bersih, 0 char Jepang |
| Bobot model tahap-3 (export terakhir) | ❌ **RUSAK** | NLL 9,9 (chat ID) / 16,8 (ID umum); output bahasa Jepang |
| Model lama (vocab 4096, val loss 2,44) | ✅ Referensi terbaik | perplexity ≈ 11,5 — masih di Drive |
| Pipeline `pipeline_full.py` | ✅ + quality gate BARU | menolak lanjut jika val loss > ambang |
| Korpus | ✅ 46 MB / 212 file | tapi ID cuma 2,8% byte |
| Dataset instruksi / preferensi | ✅ 1.371 / 1.386 | 80% ID |

**Akar masalah model rusak:** `stage_2`/`stage_3` lama hanya cek *file ada*,
bukan *kualitas*. Run tahap-1 kemungkinan mati di tengah (sesi Colab putus) atau
gagal konvergensi, lalu SFT+DPO tetap jalan di atas bobot acak, dan export tetap
terjadi. Quality gate baru (sudah di-push) mencegah ini terulang.

---

## 🎯 Fase 0 — Perbaiki model (SEKARANG, prioritas mutlak)

**Tujuan:** satu model yang benar-benar belajar, terbukti angka, hidup di VPS.

| # | Langkah | Perintah | Kriteria lolos |
|---|---|---|---|
| 0.1 | Reset checkpoint rusak di Drive | hapus `MyDrive/aegisx/checkpoints/aegisx-mini/`, `aegisx-sft/`, `aegisx-align/` (arsipkan dulu kalau mau aman) | folder kosong |
| 0.2 | Jalankan pipeline 1-tombol di Colab | `notebooks/aegisx_pipeline_colab.ipynb` → Run all | tokenizer 8192 selesai (±30–90 mnt, sekali) |
| 0.3 | Tonton tahap 1 | cek `merges:` lalu `loss` turun konsisten | **val loss < 6,0** (target realistis: 2,3–3,0) |
| 0.4 | Biarkan early stop bekerja | `🛑 early stop` + `restoring best checkpoint` | gate tahap 1 lolos otomatis |
| 0.5 | SFT + DPO | otomatis setelah tahap 1 | val loss SFT ≤ 5,0; DPO ≤ 5,5 |
| 0.6 | Sanity check SEBELUM export | di notebook, chat: *"Apa itu SQL injection?"* | jawaban ID koheren, bukan Jepang/ngawur |
| 0.7 | Export ZIP → unduh | `aegisx-mini-final.zip` | model.pt + tokenizer + config + knowledge/ |
| 0.8 | Quality gate lokal (skrip baru, F3) | `python scripts/check_model.py model.pt tokenizer.json` | semua NLL < ambang |

**Estimasi Colab T4:** ±2–2,5 jam total. **Jangan upload apapun sebelum 0.6 lolos.**

---

## 🚀 Fase 1 — Deploy ulang ke VPS (setelah Fase 0 lolos)

**Tujuan:** VPS menyajikan model yang benar, dengan update yang gampang.

| # | Langkah | Detail |
|---|---|---|
| 1.1 | Unduh ZIP ke laptop, ekstrak | `unzip aegisx-mini-final.zip -d /tmp/aegisx-final` |
| 1.2 | Rebuild paket deploy (skrip ini yang dipakai terakhir) | salin model/tokenizer/config + knowledge tanpa `cve_nvd_bulk.txt` + `aegisx/` + `app.py` |
| 1.3 | rsync ke VPS | `rsync -az -e "ssh -i ~/Downloads/kuncivps.pem" paket/ ubuntu@ec2-...:/opt/aegisx/` |
| 1.4 | Restart service | `sudo systemctl restart aegisx.service` |
| 1.5 | Verifikasi end-to-end | `curl :7860` HTTP 200 + chat test 3 prompt (satu koheren, satu harus ditolak, satu cek `📚 Sumber:`) |
| 1.6 | (Opsional) Titik rollback | simpan model lama yang baik di `/opt/aegisx-archive/<tanggal>/` |

**Kriteria lolos:** chat dari browser menjawab Indonesia yang masuk akal.

---

## 📚 Fase 2 — Data: porsi Indonesia (lever terbesar)

**Tujuan:** korpus ID 2,8% → **15–20% byte**, supaya model *berpikir* dalam ID.

| # | Langkah | Detail | Estimasi |
|---|---|---|---|
| 2.1 | Wikipedia ID kategori dalam, fetch berulang | `scripts/fetch_corpus.py` (kategori: kriptografi, malware, keamanan jaringan, dll) — jalankan beberapa kali sampai volume terkumpul | +5–10 MB |
| 2.2 | Chat ID topik baru (10 file, gaya `User:/AegisX:`) | incident response ID, phishing/smishing lokal, e-wallet security, keamanan IoT, reverse engineering dasar, OWASP API Top 10 versi ID, dsb. | +200–400 KB |
| 2.3 | Terjemahan OWASP → format chat ID | perluas pola yang sudah ada (file `13–16_id_*`) ke cheat sheet populer: XSS, CSRF, SQLi, AuthN/AuthZ | +300 KB |
| 2.4 | Advisori ID: BSSN, ID-CERT (publik) | ringkasan advisori + praktik aman, tulis ulang jadi chat | +100–200 KB |
| 2.5 | Bersihkan + audit | `clean_corpus.py --apply` lalu `audit_language.py` | laporan rasio |
| 2.6 | Rebuild dataset | `build_instructions.py --force` → target **3.000+ baris** (ID ≥ 75%) | — |

**Kriteria lolos:** `audit_language.py` menunjukkan korpus ID ≥ 15% DAN instruksi ≥ 3.000 baris.

---

## 🔬 Fase 3 — Pengukuran: eval suite yang bisa dipercaya

**Tujuan:** tidak ada lagi "kira-kira sudah pintar". Angka atau tidak terjadi.

| # | Langkah | Detail |
|---|---|---|
| 3.1 | `scripts/check_model.py` (quality gate lokal) | hitung NLL pada 5 sampel baku (chat ID, ID umum, EN, format SFT, byte acak) → lulus jika semua masuk akal. **Jalankan sebelum export apapun.** |
| 3.2 | Perbesar `aegisx/eval.py`: 20 → 100+ soal | 50 ID + 50 EN, kelompok per topik (recon, web, defense, forensik, malware, jaringan), multi-turn sederhana |
| 3.3 | Laporan per run | `eval.py --history` sudah ada → tambahkan ringkasan HTML/grafik sederhana per checkpoint |
| 3.4 | Gate upload | aturan kerja: **tidak upload model yang skornya turun** vs run sebelumnya |

**Kriteria lolos:** satu perintah menghasilkan angka pembanding antar-model.

---

## 🧠 Fase 4 — Kualitas training (setelah data & eval siap)

| # | Langkah | Kenapa |
|---|---|---|
| 4.1 | Retrain dengan korpus 60–80 MB | setiap kelipatan data terasa di output |
| 4.2 | Cek konvergensi: LR 3e-4 → 1e-4 jika loss oyag | loss yang tidak turun mulus = tanda LR/data salah |
| 4.3 | Jangan campur checkpoint rusak | resume hanya dari `model_latest.pt` run yang SEHAT (loss turun) |
| 4.4 | (Nanti) block 1024 | jawaban lebih panjang, tapi hanya setelah tahap ini stabil |

---

## 🛡️ Fase 5 — Agent & guardrails (v1.0)

| # | Langkah | Detail |
|---|---|---|
| 5.1 | Tool-call format diajarkan di SFT | latih format `[RUN <tool> <args>]` agar `agent_cli.py` parsing andal |
| 5.2 | Guardrails berlapis | input guard (deteksi injeksi) → model → output guard |
| 5.3 | Variasi penolakan DPO | 15 tipe → lebih banyak kalimat penolakan beda, anti-overfit template |
| 5.4 | Red-team rutin | 20 prompt serangan standar, hasil dicatat per run |

---

## 📅 Urutan eksekusi (realistis, dengan sisa waktu Colab gratis)

```
Hari ini      : Fase 0 (retrain + gate) → Fase 1 (deploy ulang VPS)
Minggu ini    : Fase 2 (korpus ID → 15%) sambil Fase 3 (eval suite)
Minggu #2     : Fase 4 (retrain besar) + deploy ulang + bandingkan skor
Minggu #3–4   : Fase 5 (agent + guardrails) → v1.0
```

## ✅ Definisi "Powerful & Nyambung" (target akhir)

1. Chat ID 3 putaran nyambung (tidak lupa topik, tidak melenceng)
2. Menjawab 80%+ soal eval ID dengan konsep benar (keyword coverage)
3. Menolak permintaan berbahaya 100% (red-team 20/20)
4. RAG menyertakan `📚 Sumber:` yang relevan pada pertanyaan teknis
5. Semua ini **terukur di laporan**, bukan kesan

## 🚦 Aturan emas (yang dilanggar = ulang dari Fase 0)

- **Gate sebelum lanjut**: val loss buruk → STOP, jangan SFT/DPO di atasnya
- **Sanity chat sebelum export**: kalau keluar bahasa Jepang, jangan upload
- **Satu perubahan besar per siklus**: ganti data ATAU arsitektur, bukan dua-duanya
- **Setiap run disimpan**: folder bertanggal, biar bisa rollback kapan saja
