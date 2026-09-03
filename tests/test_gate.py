import subprocess
from pathlib import Path

import pytest

from aegisx.agent import AegisAgent
from aegisx.gate import AuthorizationGate, ForbiddenCommandError, UnauthorizedTargetError


@pytest.fixture
def gate(tmp_path):
    allow = tmp_path / "authorized.txt"
    allow.write_text(
        "# authorized targets\n"
        "example.com\n"
        "*.lab.local\n"
        "192.168.1.0/24\n"
        "10.0.0.5\n",
        encoding="utf-8",
    )
    return AuthorizationGate(str(allow), str(tmp_path / "audit.log"))


def test_hostname_authorized(gate):
    assert gate.is_authorized("example.com")


def test_hostname_case_insensitive(gate):
    assert gate.is_authorized("EXAMPLE.COM")


def test_wildcard_authorized(gate):
    assert gate.is_authorized("staging.lab.local")
    assert not gate.is_authorized("lab.local")


def test_cidr_authorized(gate):
    assert gate.is_authorized("192.168.1.55")
    assert not gate.is_authorized("192.168.2.55")


def test_single_ip_authorized(gate):
    assert gate.is_authorized("10.0.0.5")
    assert not gate.is_authorized("10.0.0.6")


def test_unknown_target_denied(gate):
    assert not gate.is_authorized("evil.com")
    assert not gate.is_authorized("")
    assert not gate.is_authorized("not an ip or host")


def test_check_raises_on_unauthorized(gate):
    with pytest.raises(UnauthorizedTargetError):
        gate.check("evil.com")


def test_check_passes_on_authorized(gate):
    gate.check("example.com")  # should not raise


def test_sudo_forbidden(gate, tmp_path):
    with pytest.raises(ForbiddenCommandError):
        gate.run("test", "example.com", ["sudo", "nmap", "example.com"])


def test_shell_metacharacters_forbidden(gate, tmp_path):
    with pytest.raises(ForbiddenCommandError):
        gate.run("test", "example.com", ["echo", "a; rm -rf /"])


def test_audit_log_written(gate, tmp_path):
    gate.run("test_action", "example.com", ["true"])
    log = gate.read_log()
    assert len(log) == 1
    assert "test_action" in log[0]
    assert "example.com" in log[0]


def test_unauthorized_action_not_logged(gate, tmp_path):
    with pytest.raises(UnauthorizedTargetError):
        gate.run("test_action", "evil.com", ["true"])
    assert gate.read_log() == []


def test_run_returns_completed_process(gate, tmp_path):
    result = gate.run("test_action", "example.com", ["echo", "hello"], check_output=True)
    assert isinstance(result, subprocess.CompletedProcess)
    assert "hello" in (result.stdout or "")


def test_run_missing_binary(gate, tmp_path):
    with pytest.raises(FileNotFoundError):
        gate.run("test_action", "example.com", ["definitely-not-a-real-binary-xyz"])


def test_write_report_unauthorized_denied(gate, tmp_path):
    agent = AegisAgent(gate, reports_dir=str(tmp_path / "reports"))
    with pytest.raises(UnauthorizedTargetError):
        agent.write_report("evil.com", "t", "High", "d", "r", "i")


def test_write_report_authorized_creates_file(gate, tmp_path):
    agent = AegisAgent(gate, reports_dir=str(tmp_path / "reports"))
    path = agent.write_report("example.com", "SQL Injection", "High", "desc", "repro", "impact")
    assert Path(path).exists()
    content = Path(path).read_text(encoding="utf-8")
    assert "SQL Injection" in content
    assert "example.com" in content