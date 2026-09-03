"""AegisX agent tool layer.

Tools: recon, web, code_search, write_report — every one gated by
AuthorizationGate. Offensive tools default to read-only/scanning flags.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

from aegisx.gate import AuthorizationGate


class AegisAgent:
    def __init__(
        self,
        gate: AuthorizationGate,
        reports_dir: str | Path = "reports",
        shodan_key: Optional[str] = None,
    ) -> None:
        self.gate = gate
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.shodan_key = shodan_key

    # ------------------------------------------------------------------ #
    # Recon tools (read-only by default)
    # ------------------------------------------------------------------ #
    def recon_ports(self, target: str, ports: str = "80,443,22,8080", timeout: int = 120) -> str:
        """nmap top-service scan (no aggressive scripts)."""
        self.gate.check(target)
        result = self.gate.run(
            "recon_ports",
            target,
            ["nmap", "-Pn", "-sV", "-p", ports, target],
            timeout=timeout,
        )
        return result.stdout or result.stderr or f"nmap exit={result.returncode}"

    def recon_subdomains(self, target: str, wordlist: str, timeout: int = 120) -> str:
        """Subdomain brute force with a wordlist file (read-only DNS queries)."""
        self.gate.check(target)
        result = self.gate.run(
            "recon_subdomains",
            target,
            ["subfinder", "-d", target, "-all"],
            timeout=timeout,
        )
        return result.stdout or result.stderr or f"subfinder exit={result.returncode}"

    def recon_web_scan(self, target: str, timeout: int = 180) -> str:
        """nuclei template scan at low severity threshold (read-only checks)."""
        self.gate.check(target)
        result = self.gate.run(
            "recon_web_scan",
            target,
            ["nuclei", "-u", target, "-severity", "low,medium,high,critical", "-silent"],
            timeout=timeout,
        )
        return result.stdout or result.stderr or f"nuclei exit={result.returncode}"

    # ------------------------------------------------------------------ #
    # Web tools
    # ------------------------------------------------------------------ #
    def web_fuzz(self, target: str, wordlist: str, timeout: int = 120) -> str:
        """Directory/content fuzzing with ffuf (read-only GET requests)."""
        self.gate.check(target)
        result = self.gate.run(
            "web_fuzz",
            target,
            ["ffuf", "-u", f"{target}/FUZZ", "-w", wordlist, "-mc", "200,204,301,302,403"],
            timeout=timeout,
        )
        return result.stdout or result.stderr or f"ffuf exit={result.returncode}"

    def web_fetch(self, target: str, timeout: int = 60) -> str:
        """Fetch headers + body size of a URL (no POST, no payloads)."""
        self.gate.check(target)
        result = self.gate.run(
            "web_fetch",
            target,
            ["curl", "-sS", "-I", "-L", "--max-time", "30", target],
            timeout=timeout,
        )
        return result.stdout or result.stderr or f"curl exit={result.returncode}"

    # ------------------------------------------------------------------ #
    # Code search / SAST
    # ------------------------------------------------------------------ #
    def code_search(self, path: str, pattern: str) -> str:
        """Search a local codebase for dangerous patterns (read-only)."""
        result = self.gate.run(
            "code_search",
            "local",
            ["rg", "-n", pattern, path],
            timeout=120,
            check_output=True,
        )
        return result.stdout or result.stderr or f"rg exit={result.returncode}"

    def sast_scan(self, path: str) -> str:
        """Run semgrep static analysis on a local directory (read-only)."""
        result = self.gate.run(
            "sast_scan",
            "local",
            ["semgrep", "scan", "--quiet", path],
            timeout=300,
            check_output=True,
        )
        return result.stdout or result.stderr or f"semgrep exit={result.returncode}"

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def write_report(
        self,
        target: str,
        title: str,
        severity: str,
        description: str,
        reproduction: str,
        impact: str,
    ) -> str:
        """Write a structured bug-bounty report (markdown) to reports/."""
        self.gate.check(target)
        slug = "".join(c if c.isalnum() else "_" for c in title)[:60].lower()
        path = self.reports_dir / f"{dt.date.today().isoformat()}_{slug}.md"
        content = f"""# {title}

- **Target:** {target}
- **Severity:** {severity}
- **Date:** {dt.date.today().isoformat()}

## Summary
{description}

## Steps to Reproduce
{reproduction}

## Impact
{impact}

## Suggested Fix
- Validate and sanitize all user input server-side.
- Enforce authorization checks on every object access.
- Use parameterized queries / safe APIs; never concatenate input.
"""
        path.write_text(content, encoding="utf-8")
        self.gate.log("write_report", target, detail=str(path))
        return str(path)