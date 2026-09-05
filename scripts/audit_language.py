"""Audit bahasa: seberapa besar porsi Indonesia vs Inggris di korpus.

Deteksi berbasis stopword: menghitung kemunculan kata umum Indonesia dan
Inggris per teks, lalu mengklasifikasikan. Dipakai untuk melihat
keseimbangan EN/ID setelah tiap penambahan korpus.

Usage:
    python scripts/audit_language.py                 # audit data/raw + instructions
    python scripts/audit_language.py --raw-dir data/raw
    python scripts/audit_language.py --no-jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ID_STOPS = set("""yang dan di ke dari ini itu untuk dengan pada adalah sebagai atau tidak akan telah dapat bahwa karena namun juga agar supaya jika kalau apakah bagaimana apa siapa kapan mana kita saya mereka kami anda kamu tersebut merupakan melalui antara terhadap tentang setelah sebelum ketika sehingga sedangkan pun memang sangat paling lebih kurang saat setiap semua beberapa berbagai lain baru sudah belum pernah sedang masih harus bisa perlu mungkin hanya tanpa dalam bagi oleh secara kata contoh misalnya tetapi tetapi melainkan seperti demikian lalu maka sebab meski walaupun bila saatnya kecuali hampir sekitar segera sering kadang biasanya terutama khususnya""".split())

EN_STOPS = set("""the and of to in is are was were for with on at by from as an be this that these those it its or but not you your we our they their can will would should may must have has had do does did a into over under after before between during without about again further once here there when where why how all any both each few more most other some such no nor only own same so than too very just also because if then else could shall been being am""".split())

_WORD_RE = re.compile(r"[a-z]+")

ID_HINT = re.compile(r"\b(apa|bagaimana|mengapa|kenapa|siapa|kapan|tolong|jelaskan|sebutkan|berikan)\b", re.IGNORECASE)
EN_HINT = re.compile(r"\b(what|how|why|who|when|explain|describe|give|list)\b", re.IGNORECASE)


def detect_language(text: str) -> str:
    """Kembalikan 'id', 'en', atau 'campur' berdasarkan stopword terbanyak."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return "campur"
    id_n = sum(1 for w in words if w in ID_STOPS)
    en_n = sum(1 for w in words if w in EN_STOPS)
    if id_n > en_n * 1.25:
        return "id"
    if en_n > id_n * 1.25:
        return "en"
    return "campur"


def detect_instruction(instruction: str) -> str:
    """Deteksi bahasa instruksi: heuristik kata tanya ID vs EN."""
    if ID_HINT.search(instruction) and not EN_HINT.search(instruction):
        return "id"
    if EN_HINT.search(instruction) and not ID_HINT.search(instruction):
        return "en"
    return detect_language(instruction)


def bucket_for(name: str) -> str:
    """Kelompok file berdasarkan konvensi penamaan."""
    if "_id_chat" in name or re.match(r"^1[2-8]_|^2[0-4]_|^3[1-9]_", name):
        return "chat/artikel ID"
    if "_en_chat" in name or re.match(r"^0[1-9]_|^1[0-1]_|^2[5-8]_", name):
        return "prosa/chat EN"
    if name.startswith("cve_") or name.startswith("owasp_") or name.startswith("mitre_") or name.startswith("payloads_"):
        return "sumber publik (OWASP/MITRE/CVE)"
    if name.startswith("id_wikipedia"):
        return "Wikipedia ID"
    return "lainnya"


def fmt_bytes(b: int) -> str:
    if b >= 1024 * 1024:
        return f"{b / 1024 / 1024:.1f} MB"
    return f"{b / 1024:.0f} KB"


def audit_raw(raw_dir: Path) -> None:
    print("=" * 62)
    print(f"KORPUS  ({raw_dir})")
    print("=" * 62)
    files = sorted(raw_dir.glob("*.txt"))
    per_bucket: dict[str, dict[str, int]] = {}
    per_lang: dict[str, int] = {"id": 0, "en": 0, "campur": 0}
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        size = len(text)
        lang = detect_language(text)
        per_lang[lang] += size
        bk = per_bucket.setdefault(bucket_for(f.name), {"id": 0, "en": 0, "campur": 0})
        bk[lang] += size
    total = sum(per_lang.values())
    print(f"{'Kelompok':<36}{'ID':>10}{'EN':>10}{'Campur':>10}")
    print("-" * 62)
    for bk in sorted(per_bucket):
        d = per_bucket[bk]
        print(f"{bk:<36}{fmt_bytes(d['id']):>10}{fmt_bytes(d['en']):>10}{fmt_bytes(d['campur']):>10}")
    print("-" * 62)
    print(f"{'TOTAL':<36}{fmt_bytes(per_lang['id']):>10}{fmt_bytes(per_lang['en']):>10}{fmt_bytes(per_lang['campur']):>10}")
    print(f"\nProporsi (bytes):  ID {per_lang['id'] / total * 100:.1f}% | "
          f"EN {per_lang['en'] / total * 100:.1f}% | campur {per_lang['campur'] / total * 100:.1f}%")


def audit_jsonl(jsonl: Path) -> None:
    print("\n" + "=" * 62)
    print(f"DATASET INSTRUKSI  ({jsonl.name})")
    print("=" * 62)
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    id_n = en_n = mix_n = 0
    lang_example: dict[str, str] = {}
    for r in rows:
        lang = detect_instruction(r.get("instruction", ""))
        if lang == "id":
            id_n += 1
        elif lang == "en":
            en_n += 1
        else:
            mix_n += 1
        lang_example.setdefault(lang, r.get("instruction", "")[:70])
    total = len(rows)
    print(f"total baris : {total}")
    print(f"ID          : {id_n} ({id_n / total * 100:.1f}%)")
    print(f"EN          : {en_n} ({en_n / total * 100:.1f}%)")
    print(f"campur/ambig: {mix_n}")
    print("\ncontoh per bahasa:")
    for lang, ex in lang_example.items():
        print(f"  [{lang}] {ex}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--jsonl", default="data/finetune/instructions.jsonl")
    parser.add_argument("--no-jsonl", action="store_true")
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        print(f"Folder korpus tidak ditemukan: {raw_dir}", file=sys.stderr)
        sys.exit(1)
    audit_raw(raw_dir)
    if not args.no_jsonl:
        jsonl = Path(args.jsonl)
        if jsonl.exists():
            audit_jsonl(jsonl)


if __name__ == "__main__":
    sys.exit(main())
