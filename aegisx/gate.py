"""Authorization gate for AegisX agent actions.

Every offensive action must target a host on the authorized allowlist.
All actions are written to an append-only audit log. No sudo, no shell=True,
no out-of-scope targets.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import subprocess
from pathlib import Path
from typing import Iterable, Optional


class UnauthorizedTargetError(PermissionError):
    """Raised when an action targets a host outside the allowlist."""


class ForbiddenCommandError(PermissionError):
    """Raised when a command violates the gate rules (sudo, shell=True)."""


class AuthorizationGate:
    def __init__(self, allowlist_path: str | Path, audit_log_path: str | Path) -> None:
        self.allowlist_path = Path(allowlist_path)
        self.audit_log_path = Path(audit_log_path)
        self._allowlist: set[str] = set()
        self._cidr: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        if self.allowlist_path.exists():
            self._load_allowlist(self.allowlist_path)

    # ------------------------------------------------------------------ #
    # Allowlist
    # ------------------------------------------------------------------ #
    def _load_allowlist(self, path: Path) -> None:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                net = ipaddress.ip_network(line, strict=False)
                self._cidr.append(net)
            except ValueError:
                self._allowlist.add(line.lower())

    def add_allowlist(self, path: str | Path) -> None:
        """Add entries from another file at runtime."""
        self._load_allowlist(Path(path))

    def is_authorized(self, target: str) -> bool:
        target = target.strip().lower()
        if not target:
            return False
        # Exact hostname match (supports *.example.com wildcards).
        for entry in self._allowlist:
            if entry.startswith("*."):
                if target.endswith(entry[1:]):
                    return True
            elif target == entry:
                return True
        # IP / CIDR match.
        try:
            ip = ipaddress.ip_address(target)
        except ValueError:
            return False
        return any(ip in net for net in self._cidr)

    # ------------------------------------------------------------------ #
    # Audit log
    # ------------------------------------------------------------------ #
    def log(self, action: str, target: str, detail: str = "") -> None:
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        line = f"{ts} | action={action} | target={target} | {detail}\n"
        with self.audit_log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def read_log(self) -> list[str]:
        if not self.audit_log_path.exists():
            return []
        return self.audit_log_path.read_text(encoding="utf-8").splitlines()

    # ------------------------------------------------------------------ #
    # Command execution
    # ------------------------------------------------------------------ #
    def check(self, target: str) -> None:
        """Validate target before any action. Raises if unauthorized."""
        if not self.is_authorized(target):
            raise UnauthorizedTargetError(
                f"Target '{target}' is not on the authorized allowlist. "
                f"Add it to {self.allowlist_path} first. Bug bounty = only in-scope targets."
            )

    def run(
        self,
        action: str,
        target: str,
        argv: list[str],
        timeout: int = 60,
        check_output: bool = False,
    ) -> subprocess.CompletedProcess:
        """Check authorization, audit, and execute argv (no shell).

        Raises UnauthorizedTargetError / ForbiddenCommandError / FileNotFoundError.
        """
        self.check(target)
        if any(token == "sudo" or token.startswith("sudo ") for token in argv):
            raise ForbiddenCommandError("sudo is forbidden in AegisX agent actions.")
        if any("|" in t or ";" in t or ">" in t or "$(" in t for t in argv):
            raise ForbiddenCommandError("Shell metacharacters are forbidden; pass argv directly.")

        self.log(action, target, detail=" ".join(argv))
        return subprocess.run(
            argv,
            capture_output=check_output,
            text=check_output,
            timeout=timeout,
            check=False,
        )