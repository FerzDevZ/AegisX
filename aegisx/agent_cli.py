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

from aegisx.agent import AegisAgent
from aegisx.chat import DEFAULT_PROMPT, generate
from aegisx.gate import AuthorizationGate

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
    max_tool_rounds: int = 3,
    **gen_kwargs,
) -> str:
    """One user turn with up to max_tool_rounds tool executions."""
    conversation = history
    for _ in range(max_tool_rounds):
        prompt = DEFAULT_PROMPT + user_input + "\n\n" + conversation + "AegisX: "
        raw = generate(model_path, tokenizer_path, prompt, **gen_kwargs)

        call = extract_tool_call(raw)
        if call is None:
            conversation += f"AegisX: {raw}\n"
            return conversation

        tool_name, target = call
        ok, result = execute_tool(agent, tool_name, target)
        if not ok:
            conversation += f"AegisX: {raw}\n(agent note: {result})\n"
            return conversation
        # Feed tool output back so the model can summarize it.
        user_input = user_input
        conversation += (
            f"AegisX: {raw}\n"
            f"[tool {tool_name} output]\n{result[:2000]}\n[/tool]\n"
        )
    return conversation + "\n(stopped: tool round limit reached)\n"


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

    print("AegisX agent mode — @tool recon_ports <target> to run gated recon. Ctrl+C / 'exit' to quit.")
    history = ""
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
        history = run_turn(args.model, args.tokenizer, agent, user, history=history, **kwargs)
        # Print just the last assistant message.
        last = history.strip().split("\nAegisX: ")[-1]
        print(f"\nAegisX> {last}")


if __name__ == "__main__":
    sys.exit(main())