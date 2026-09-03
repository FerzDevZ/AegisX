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

CS_BASE = "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/cheatsheets/"

# OWASP CheatSheetSeries files (markdown) that train well as plain text.
OWASP_CHEATSHEETS: dict[str, str] = {
    f"owasp_{name}.txt": f"{CS_BASE}{filename}.md"
    for name, filename in {
        "sqli": "SQL_Injection_Prevention_Cheat_Sheet",
        "xss": "Cross_Site_Scripting_Prevention_Cheat_Sheet",
        "csrf": "Cross-Site_Request_Forgery_Prevention_Cheat_Sheet",
        "auth": "Authentication_Cheat_Sheet",
        "password_storage": "Password_Storage_Cheat_Sheet",
        "injection": "Injection_Prevention_Cheat_Sheet",
        "ssrf": "Server_Side_Request_Forgery_Prevention_Cheat_Sheet",
        "deserialization": "Deserialization_Cheat_Sheet",
        "file_upload": "File_Upload_Cheat_Sheet",
        "http_headers": "HTTP_Headers_Cheat_Sheet",
        "transport_protection": "Transport_Layer_Protection_Cheat_Sheet",
        "logging": "Logging_Cheat_Sheet",
        "rest": "REST_Security_Cheat_Sheet",
        "oauth": "OAuth2_Cheat_Sheet",
        "jwt": "JSON_Web_Token_Cheat_Sheet",
        "access_control": "Access_Control_Cheat_Sheet",
        "cryptographic_storage": "Cryptographic_Storage_Cheat_Sheet",
        "input_validation": "Input_Validation_Cheat_Sheet",
        "mass_assignment": "Mass_Assignment_Cheat_Sheet",
        "clickjacking": "Clickjacking_Defense_Cheat_Sheet",
        "session_management": "Session_Management_Cheat_Sheet",
        "tls": "TLS_Cipher_String_Cheat_Sheet",
        "pinning": "Pinning_Cheat_Sheet",
        "secrets": "Secrets_Management_Cheat_Sheet",
        "api_auth": "Authorization_Testing_Automation_Cheat_Sheet",
        "html5": "HTML5_Security_Cheat_Sheet",
        "dotnet": "DotNet_Security_Cheat_Sheet",
        "nodejs": "Nodejs_Security_Cheat_Sheet",
        "php": "PHP_Configuration_Cheat_Sheet",
        "java": "Java_Security_Cheat_Sheet",
    }.items()
}

SOURCES: dict[str, str] = {
    **OWASP_CHEATSHEETS,
    # OWASP ASVS 5.0 — flat JSON holds every requirement (name + description)
    "owasp_asvs.txt": (
        "https://raw.githubusercontent.com/OWASP/ASVS/master/5.0/docs_en/"
        "OWASP_Application_Security_Verification_Standard_5.0.0_en.flat.json"
    ),
    # MITRE ATT&CK enterprise techniques (STIX data, public knowledge base)
    "mitre_attck.txt": (
        "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
        "enterprise-attack/enterprise-attack.json"
    ),
}

# Real, fetchable CVE IDs to include (descriptions give vuln-pattern vocabulary).
CVE_SAMPLE_IDS = [
    "CVE-2021-44228",  # Log4Shell
    "CVE-2017-0144",   # EternalBlue
    "CVE-2023-44487",  # HTTP/2 Rapid Reset
    "CVE-2022-22965",  # Spring4Shell
    "CVE-2023-23397",  # Outlook
    "CVE-2024-3400",   # PAN-OS
]


MITRE_CVE_API = "https://cveawg.mitre.org/api/cve/"


def fetch_cve(url: str, dest: Path) -> bool:
    """Fetch a MITRE CVE record (JSON) and flatten it to readable text."""
    import json

    print(f"  ↓ {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"    ✗ failed: {exc}")
        return False

    lines = [f"CVE: {data.get('cveMetadata', {}).get('cveId', url.split('/')[-1])}"]
    containers = data.get("containers", {}).get("cna", {})
    if containers.get("title"):
        lines.append(f"Title: {containers['title']}")
    for desc in containers.get("descriptions", []):
        if desc.get("lang") == "en":
            lines.append(f"Description: {desc.get('value', '')}")
    for metric in containers.get("metrics", [])[:1]:
        cvss = metric.get("cvssV3_1", {})
        if cvss:
            lines.append(f"Severity: {cvss.get('baseSeverity', '')} ({cvss.get('baseScore', '')})")
            lines.append(f"Vector: {cvss.get('vectorString', '')}")
    text = "\n".join(lines).strip()
    if not text:
        return False
    dest.write_text(text, encoding="utf-8")
    print(f"    ✓ {len(text):,} chars -> {dest.name}")
    return True


def fetch(url: str, dest: Path) -> bool:
    print(f"  ↓ {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            body = resp.read()
    except Exception as exc:  # network errors, 404s, etc.
        print(f"    ✗ failed: {exc}")
        return False
    text = body.decode("utf-8", errors="replace")
    # JSON sources (ASVS flat, MITRE STIX) are huge; flatten to plain text.
    if url.rstrip().lower().endswith(".json"):
        try:
            text = flatten_json(text)
        except Exception as exc:
            print(f"    ✗ json parse failed: {exc}")
            return False
    dest.write_text(text, encoding="utf-8")
    print(f"    ✓ {len(text):,} chars -> {dest.name}")
    return True


def flatten_json(raw: str) -> str:
    """Turn ASVS/MITRE JSON into readable plain text for training."""
    import json
    import re

    def clean(text: str) -> str:
        # Strip markdown link syntax: [label](url) -> label
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"[\n\r]+", " ", text)
        return text.strip()

    data = json.loads(raw)
    lines: list[str] = []
    if isinstance(data, dict) and isinstance(data.get("requirements"), list):
        # OWASP ASVS flat JSON: [{req_id, req_description, section_name, ...}]
        for req in data["requirements"]:
            rid = req.get("req_id") or ""
            desc = clean(req.get("req_description") or "")
            if desc:
                lines.append(f"Requirement {rid}: {desc}")
        return "\n\n".join(lines)

    # MITRE ATT&CK STIX bundle: objects[] with type/name/description.
    objs = data.get("objects", []) if isinstance(data, dict) else data
    for item in objs:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {"attack-pattern", "x-mitre-tactic", "tool", "malware"}:
            continue  # drop relationships, groups, campaigns — keep techniques
        name = clean(item.get("name") or "")
        desc = clean(item.get("description") or "")
        if name and desc:
            lines.append(f"{item.get('type')}: {name}. {desc}")
    return "\n\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public cysec corpora")
    parser.add_argument("--target-dir", default="data/raw")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    out_dir = Path(args.target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    fetched: dict[str, str] = dict(SOURCES)

    # Add individual CVE records (JSON -> flatten to text).
    for cve_id in CVE_SAMPLE_IDS:
        fetched[f"cve_{cve_id}.txt"] = f"{MITRE_CVE_API}{cve_id}"

    for name, url in fetched.items():
        dest = out_dir / name
        if dest.exists() and not args.force:
            print(f"  = {name} exists, skipping (--force to re-download)")
            ok += 1
            continue
        if url.startswith(MITRE_CVE_API):
            if fetch_cve(url, dest):
                ok += 1
        else:
            if fetch(url, dest):
                ok += 1

    total = sum(p.stat().st_size for p in out_dir.glob("*.txt"))
    print(f"\nDone: {ok}/{len(SOURCES)} sources. Corpus now ~{total / 1024:.0f} KB.")
    if total < 200 * 1024:
        print("Tip: add your own .txt files too — more relevant text = better model.")


if __name__ == "__main__":
    sys.exit(main())