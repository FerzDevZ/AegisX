import pytest

from aegisx.agent import AegisAgent
from aegisx.agent_cli import execute_tool, extract_tool_call, run_turn
from aegisx.gate import AuthorizationGate


@pytest.fixture
def gate(tmp_path):
    allow = tmp_path / "authorized.txt"
    allow.write_text("example.com\n192.168.1.0/24\n", encoding="utf-8")
    return AuthorizationGate(str(allow), str(tmp_path / "audit.log"))


def test_extract_tool_call_found():
    text = "I will scan now.\n@tool recon_ports example.com\nThen summarize."
    assert extract_tool_call(text) == ("recon_ports", "example.com")


def test_extract_tool_call_none():
    assert extract_tool_call("Just a normal answer.") is None
    assert extract_tool_call("") is None


def test_extract_tool_call_case_insensitive():
    text = "@TOOL WEB_FETCH Example.COM"
    assert extract_tool_call(text) == ("web_fetch", "Example.COM")


def test_execute_unknown_tool(gate):
    agent = AegisAgent(gate)
    ok, msg = execute_tool(agent, "nonsense", "example.com")
    assert not ok
    assert "Unknown tool" in msg


def test_execute_unauthorized_blocked(gate):
    agent = AegisAgent(gate)
    ok, msg = execute_tool(agent, "recon_ports", "evil.com")
    assert not ok
    assert "BLOCKED" in msg
    assert "authorization" in msg.lower()


def test_execute_missing_binary_graceful(gate, monkeypatch):
    # Simulate the tool binary not being installed (deterministic).
    agent = AegisAgent(gate)

    def boom(*a, **k):
        raise FileNotFoundError("nmap")

    monkeypatch.setattr(agent, "recon_ports", boom)
    ok, msg = execute_tool(agent, "recon_ports", "example.com")
    assert not ok
    assert "not installed" in msg


def test_run_turn_no_tool_with_fake_model(tmp_path, gate, monkeypatch):
    """Model that returns plain text -> no tool execution."""
    import aegisx.agent_cli as mod

    agent = AegisAgent(gate)
    monkeypatch.setattr(mod, "generate", lambda *a, **k: "Here is a plain answer.")
    history = run_turn("model.pt", "tok.json", agent, "explain xss", history="")
    assert "Here is a plain answer." in history
    assert gate.read_log() == []  # no tool was run


def test_run_turn_tool_round_with_fake_model(gate, monkeypatch):
    """Model emits a tool call; output is fed back; unauthorized target blocked."""
    import aegisx.agent_cli as mod

    agent = AegisAgent(gate)

    def fake_generate(*a, **k):
        return "@tool recon_ports evil.com"

    monkeypatch.setattr(mod, "generate", fake_generate)
    history = run_turn("model.pt", "tok.json", agent, "scan for me")
    assert "BLOCKED" in history
    assert "authorization gate" in history