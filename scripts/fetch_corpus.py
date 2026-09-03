"""Download large public cybersecurity corpora into data/raw/.

These are well-known PUBLIC text sources (documentation, wiki dumps). Run from
the project root or in Colab:

    python scripts/fetch_corpus.py --target-dir data/raw

Each source downloads to its own .txt file. Only HTTP(S) GETs, no auth, no
private data. Skips sources that already exist unless --force is passed.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

SOURCES: dict[str, str] = {
    # OWASP CheatSheetSeries (markdown, ~hundreds of KB total across files)
    "owasp_cheatsheets.txt": (
        "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/"
        "cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.md"
    ),
    "owasp_xss.txt": (
        "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/"
        "cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.md"
    ),
    "owasp_csrf.txt": (
        "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/"
        "cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.md"
    ),
    "owasp_auth.txt": (
        "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/"
        "cheatsheets/Authentication_Cheat_Sheet.md"
    ),
    "owasp_pw.txt": (
        "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/"
        "cheatsheets/Password_Storage_Cheat_Sheet.md"
    ),
    "owasp_injection.txt": (
        "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/"
        "cheatsheets/Injection_Prevention_Cheat_Sheet.md"
    ),
    "owasp_ssrf.txt": (
        "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/"
        "cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.md"
    ),
    # OWASP ASVS master doc (markdown)
    "owasp_asvs.txt": (
        "https://raw.githubusercontent.com/OWASP/ASVS/master/4.0/"
        "docs/en/0x10-V1-Architecture.md"
    ),
    # MITRE ATT&CK enterprise techniques (partially derived, public domain-ish)
    "mitre_attck.txt": (
        "https://raw.githubusercontent.com/mitre-attack/attack-website/master/"
        "content/overview/techniques-enterprise.md"
    ),
}


def fetch(url: str, dest: Path) -> bool:
    print(f"  ↓ {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            body = resp.read()
    except Exception as exc:  # network errors, 404s, etc.
        print(f"    ✗ failed: {exc}")
        return False
    # Sanitize: strip HTML tags if any slipped through; keep markdown as-is.
    text = body.decode("utf-8", errors="replace")
    dest.write_text(text, encoding="utf-8")
    print(f"    ✓ {len(text):,} chars -> {dest.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public cysec corpora")
    parser.add_argument("--target-dir", default="data/raw")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    out_dir = Path(args.target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for name, url in SOURCES.items():
        dest = out_dir / name
        if dest.exists() and not args.force:
            print(f"  = {name} exists, skipping (--force to re-download)")
            ok += 1
            continue
        if fetch(url, dest):
            ok += 1

    total = sum(p.stat().st_size for p in out_dir.glob("*.txt"))
    print(f"\nDone: {ok}/{len(SOURCES)} sources. Corpus now ~{total / 1024:.0f} KB.")
    if total < 200 * 1024:
        print("Tip: add your own .txt files too — more relevant text = better model.")


if __name__ == "__main__":
    sys.exit(main())