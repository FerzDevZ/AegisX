"""Agentic chat loop for AegisX.

The model generates text; if its output contains a tool directive on its own
line (e.g. `@tool recon_ports example.com`), the directive is executed through
the authorization gate and the result is fed back into the conversation.
Anything without a directive is a plain answer.

Usage:
    python -m aegisx.agent_cli \
        --model checkpoints/aegisx-mini/model.pt \
        --tokenizer checkpoints/aegisx-mini/tokenizer.json \
        --allowlist targets/authorized.txt --audit-log logs/audit.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable

from aegisx.agent import AegisAgent
from aegisx.chat import DEFAULT_PROMPT, generate
from aegisx.gate import AuthorizationGate

# Harness note fed back after every tool result (ReAct reflection step).
_REFLECT_NOTE = (
    "[system] Evaluasi hasil tool di atas. Kalau sudah cukup untuk menjawab "
    "pengguna, jawab sekarang tanpa memanggil tool lagi. Kalau perlu data "
    "tambahan, panggil satu @tool lagi yang paling relevan."
)

_TOOL_LINE = re.compile(r"^@tool\s+(\w+)\s+(\S+)\s*$", re.IGNORECASE)

# Tool name -> agent method (read-only / gated).
_TOOL_METHODS: dict[str, str] = {
    "recon_ports": "recon_ports",
    "recon_web": "recon_web_scan",
    "web_fetch": "web_fetch",
}


def extract_tool_call(text: str) -> tuple[str, str] | None:
    """Return (tool_name, target) for the first @tool directive line, else None."""
    for line in text.splitlines():
        m = _TOOL_LINE.match(line.strip())
        if m:
            return m.group(1).lower(), m.group(2)
    return None


def execute_tool(
    agent: AegisAgent, tool_name: str, target: str
) -> tuple[bool, str]:
    """Run a gated tool. Returns (ok, output_or_error)."""
    if tool_name not in _TOOL_METHODS:
        return False, f"Unknown tool '{tool_name}'. Known: {sorted(_TOOL_METHODS)}"
    try:
        method = getattr(agent, _TOOL_METHODS[tool_name])
        out = method(target)
        return True, (out or f"({tool_name} returned no output)")
    except PermissionError as exc:
        return False, f"BLOCKED by authorization gate: {exc}"
    except FileNotFoundError:
        return False, f"Tool binary not installed locally ({tool_name}); add it to PATH or run in Colab."


def run_turn(
    model_path: str,
    tokenizer_path: str,
    agent: AegisAgent,
    user_input: str,
    history: str = "",
    max_tool_rounds: int = 5,
    confirm_fn: Callable[[str, str], bool] | None = None,
    **gen_kwargs,
) -> str:
    """One ReAct turn: plan -> act (gated tool) -> reflect, up to N rounds.

    confirm_fn (optional human-in-the-loop gate): called as
    confirm_fn(tool_name, target) before every execution; returning False
    skips the tool. The harness appends a reflection note after each tool
    result so the model decides: answer now, or call another tool.
    """
    conversation = history
    for _ in range(max_tool_rounds):
        prompt = DEFAULT_PROMPT + user_input + "\n\n" + conversation + "AegisX: "
        raw = generate(model_path, tokenizer_path, prompt, **gen_kwargs)

        call = extract_tool_call(raw)
        if call is None:
            conversation += f"AegisX: {raw}\n"
            return conversation

        tool_name, target = call
        if confirm_fn is not None and not confirm_fn(tool_name, target):
            conversation += f"AegisX: {raw}\n(agent note: user declined to run {tool_name} on {target}; answer from what you already know.)\n"
            return conversation

        ok, result = execute_tool(agent, tool_name, target)
        if not ok:
            conversation += f"AegisX: {raw}\n(agent note: {result})\n"
            return conversation
        # Feed tool output + reflection note back so the loop can continue.
        conversation += (
            f"AegisX: {raw}\n"
            f"[tool {tool_name} output]\n{result[:2000]}\n[/tool]\n"
            f"{_REFLECT_NOTE}\n"
        )
    return conversation + "\n(stopped: tool round limit reached)\n"


# --------------------------------------------------------------------- #
# Session persistence: survive disconnects, same philosophy as training. #
# --------------------------------------------------------------------- #
def save_session(path: str, history: str) -> None:
    """Append the current conversation state to a SQLite session file."""
    import sqlite3
    import time

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS turns ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, history TEXT NOT NULL)"
        )
        con.execute("INSERT INTO turns (ts, history) VALUES (?, ?)", (time.time(), history))
        con.commit()
    finally:
        con.close()


def load_session(path: str) -> str:
    """Return the newest saved conversation state, or '' when none exists."""
    import sqlite3

    p = Path(path)
    if not p.exists():
        return ""
    con = sqlite3.connect(str(p))
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS turns ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, history TEXT NOT NULL)"
        )
        row = con.execute("SELECT history FROM turns ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else ""
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="AegisX agentic chat")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--allowlist", default="targets/authorized.txt")
    parser.add_argument("--audit-log", default="logs/audit.log")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--max-tool-rounds", type=int, default=5, help="max tool rounds per user turn (ReAct loop cap)")
    parser.add_argument("--confirm", action="store_true", help="ask y/N before every tool execution (human-in-the-loop)")
    parser.add_argument("--session", default="logs/agent_session.db", help="SQLite session state file")
    parser.add_argument("--resume", action="store_true", help="continue from the last saved session")
    args = parser.parse_args()

    gate = AuthorizationGate(args.allowlist, args.audit_log)
    agent = AegisAgent(gate)
    kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=1.1,
        device=args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
    )

    confirm_fn: Callable[[str, str], bool] | None = None
    if args.confirm:
        def confirm_fn(tool_name: str, target: str) -> bool:  # noqa: E306
            return input(f"  [confirm] run {tool_name} on {target}? [y/N] ").strip().lower().startswith("y")

    print("AegisX agent mode (ReAct, max %d rounds) — @tool <name> <target>. Ctrl+C / 'exit' to quit." % args.max_tool_rounds)
    history = load_session(args.session) if args.resume else ""
    if history:
        print(f"(session resumed from {args.session})")
    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break
        history = run_turn(
            args.model,
            args.tokenizer,
            agent,
            user,
            history=history,
            max_tool_rounds=args.max_tool_rounds,
            confirm_fn=confirm_fn,
            **kwargs,
        )
        save_session(args.session, history)
        # Print just the last assistant message.
        last = history.strip().split("\nAegisX: ")[-1]
        print(f"\nAegisX> {last}")


if __name__ == "__main__":
    sys.exit(main())