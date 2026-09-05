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
    # MITRE ATT&CK domains (STIX data, public knowledge base)
    "mitre_attck.txt": (
        "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
        "enterprise-attack/enterprise-attack.json"
    ),
    "mitre_attck_ics.txt": (
        "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
        "ics-attack/ics-attack.json"
    ),
    "mitre_attck_mobile.txt": (
        "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
        "mobile-attack/mobile-attack.json"
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


def fetch_tarball_md(tarball_url: str, dest: Path) -> bool:
    """Download a GitHub codeload tarball and concat every .md into one .txt."""
    import tarfile
    import io

    print(f"  ↓ {tarball_url}")
    try:
        with urllib.request.urlopen(tarball_url, timeout=120) as resp:
            body = resp.read()
    except Exception as exc:
        print(f"    ✗ download failed: {exc}")
        return False
    try:
        tf = tarfile.open(fileobj=io.BytesIO(body), mode="r:gz")
    except Exception as exc:
        print(f"    ✗ not a gzip tarball: {exc}")
        return False
    docs = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".md")]
    chunks = []
    for m in sorted(docs, key=lambda x: x.name):
        raw = tf.extractfile(m).read().decode("utf-8", errors="replace")
        if len(raw) >= 300:
            chunks.append(raw.strip())
    if not chunks:
        print("    ✗ no markdown found")
        return False
    dest.write_text("\n\n".join(chunks), encoding="utf-8")
    print(f"    ✓ {len(chunks)} docs -> {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
    return True


CSS_API = "https://api.github.com/repos/OWASP/CheatSheetSeries/contents/cheatsheets"


def fetch_all_owasp_cheatsheets(out_dir: Path, force: bool = False) -> int:
    """Fetch every CheatSheetSeries sheet not already in out_dir."""
    import json

    req = urllib.request.Request(CSS_API, headers={"User-Agent": "aegisx-corpus"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            items = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"  ✗ list cheatsheets failed: {exc}")
        return 0
    fetched = 0
    for it in items:
        name = it.get("name", "")
        if not name.endswith(".md") or "_Index" in name:
            continue
        dest = out_dir / f"owasp_{name[:-3].lower()}.txt"
        if dest.exists() and not force:
            continue
        try:
            with urllib.request.urlopen(it["download_url"], timeout=60) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except Exception:
            continue
        dest.write_text(body, encoding="utf-8")
        fetched += 1
    print(f"  = OWASP CheatSheetSeries: {fetched} sheet baru (total {len(items) - 1})")
    return fetched


NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def fetch_nvd_bulk(dest: Path, max_pages: int = 5, days: int = 120, start_index: int = 0) -> bool:
    """Pull CVE records from the NVD API (thousands of descriptions).

    NVD is rate-limited without an API key (~5 req/30s), so we page slowly.
    Each page returns up to 2000 CVEs; max_pages defaults to 5 = up to ~10k
    CVEs. Restricts to the last `days` days (120 = NVD no-key maximum for a
    date range) so recent runs hold modern, relevant CVEs; pass days=0 and a
    start_index to walk the full database in chunks. Writes plain text:
    id + description + severity.
    """
    import datetime
    import json
    import time
    import urllib.parse

    written = 0
    per_page = 2000
    if days > 0:
        # Recent window: modern CVE descriptions are richer and more relevant.
        now = datetime.datetime.now(datetime.timezone.utc)
        start = now - datetime.timedelta(days=days)
        date_q = (
            f"&pubStartDate={urllib.parse.quote(start.strftime('%Y-%m-%dT%H:%M:%S.000') + 'Z')}"
            f"&pubEndDate={urllib.parse.quote(now.strftime('%Y-%m-%dT%H:%M:%S.000') + 'Z')}"
        )
    else:
        date_q = ""
    time.sleep(1.0)
    for page in range(max_pages):
        url = f"{NVD_API}?resultsPerPage={per_page}&startIndex={start_index}{date_q}"
        print(f"  ↓ NVD page {page + 1}/{max_pages} (startIndex={start_index})")
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"    ✗ NVD page {page + 1} failed: {exc}")
            break
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            break
        lines: list[str] = []
        for item in vulns:
            cve = item.get("cve", {})
            cve_id = cve.get("id", "")
            desc = ""
            for d in cve.get("descriptions", []):
                if d.get("lang") == "en":
                    desc = d.get("value", "").strip()
                    break
            severity = ""
            for m in cve.get("metrics", {}).values():
                if m:
                    cvss = m[0].get("cvssData", {})
                    severity = cvss.get("baseSeverity", "")
                    break
            if cve_id and desc:
                sev = f" Severity: {severity}." if severity else ""
                lines.append(f"CVE: {cve_id}.{sev} {desc}")
        if not lines:
            break
        with dest.open("a", encoding="utf-8") as fh:
            fh.write("\n\n".join(lines) + "\n")
        written += len(lines)
        total_results = data.get("totalResults", 0)
        start_index += per_page
        if start_index >= total_results:
            break
        time.sleep(7.0)  # respect NVD rate limit without an API key
    print(f"    ✓ NVD bulk: {written:,} CVE descriptions -> {dest.name}")
    return written > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public cysec corpora")
    parser.add_argument("--target-dir", default="data/raw")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--cve-pages", type=int, default=5, help="NVD API pages to fetch (0 = skip NVD bulk)")
    parser.add_argument("--cve-days", type=int, default=120, help="NVD window in days (0 = whole database, walk with --cve-start-index)")
    parser.add_argument("--cve-start-index", type=int, default=0, help="NVD offset for chunked full-database walks")
    parser.add_argument("--cve-keywords", default="", help="optional NVD keyword filter, e.g. 'sql injection'")
    args = parser.parse_args()

    out_dir = Path(args.target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    fetched: dict[str, str] = dict(SOURCES)

    # Bulk NVD fetch into one growing file. --cve-start-index lets you walk
    # the whole (public) database in chunks: each run appends the next slice.
    if args.cve_pages > 0:
        nvd_dest = out_dir / "cve_nvd_bulk.txt"
        start_at_zero = args.cve_start_index == 0
        if nvd_dest.exists() and start_at_zero and not args.force:
            print(f"  = cve_nvd_bulk.txt exists ({nvd_dest.stat().st_size / 1024:.0f} KB), skipping (--force to re-download)")
        else:
            if nvd_dest.exists() and start_at_zero:
                nvd_dest.unlink()  # force + fresh start => rebuild whole file
            if fetch_nvd_bulk(nvd_dest, max_pages=args.cve_pages, days=args.cve_days, start_index=args.cve_start_index):
                ok += 1

    # Optional: whole OWASP CheatSheetSeries (adds any sheet not already saved).
    fetch_all_owasp_cheatsheets(out_dir, force=args.force)

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

    # Guide-sized markdown repos: WSTG + PayloadsAllTheThings.
    md_sources = [
        ("owasp_wstg.txt", "https://codeload.github.com/OWASP/wstg/tar.gz/refs/heads/master"),
        ("payloads_all_the_things.txt", "https://codeload.github.com/swisskyrepo/PayloadsAllTheThings/tar.gz/refs/heads/master"),
    ]
    for name, url in md_sources:
        dest = out_dir / name
        if dest.exists() and not args.force:
            print(f"  = {name} exists, skipping (--force to re-download)")
            continue
        if fetch_tarball_md(url, dest):
            ok += 1

    total = sum(p.stat().st_size for p in out_dir.glob("*.txt"))
    print(f"\nDone: {ok}/{len(SOURCES) + (1 if args.cve_pages > 0 else 0)} sources. Corpus now ~{total / 1024:.0f} KB.")
    if total < 200 * 1024:
        print("Tip: add your own .txt files too — more relevant text = better model.")


if __name__ == "__main__":
    sys.exit(main())