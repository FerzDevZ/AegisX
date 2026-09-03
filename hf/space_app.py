"""AegisX-Mini Gradio chat app for Hugging Face Spaces.

Two supported layouts:
  1. Model files INSIDE the Space (model.pt + tokenizer.json at repo root)
     -> no env vars needed, simplest possible setup.
  2. Model on the Hub: set AEGISX_REPO env var to your model repo id.

Environment variables (optional):
    AEGISX_REPO   e.g. youruser/aegisx-mini  (only needed for layout 2)
    AEGISX_PROMPT system prompt prefix
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from aegisx.chat import generate

CACHE = Path("/tmp/aegisx-model")
CACHE.mkdir(parents=True, exist_ok=True)

# Layout 1: model files shipped inside the Space repo root.
LOCAL_MODEL = Path(__file__).resolve().parent / "model.pt"
LOCAL_TOKENIZER = Path(__file__).resolve().parent / "tokenizer.json"

REPO = os.environ.get("AEGISX_REPO", "")
if LOCAL_MODEL.exists() and LOCAL_TOKENIZER.exists():
    MODEL_PATH = str(LOCAL_MODEL)
    TOKENIZER_PATH = str(LOCAL_TOKENIZER)
    print("Using model files shipped inside the Space.")
else:
    # Layout 2: fetch from a model repo on the Hub.
    from huggingface_hub import snapshot_download

    if not REPO:
        raise SystemExit("No local model.pt found and AEGISX_REPO env var is not set.")
    snapshot_download(repo_id=REPO, local_dir=str(CACHE))
    MODEL_PATH = str(CACHE / "model.pt")
    TOKENIZER_PATH = str(CACHE / "tokenizer.json")
    print(f"Downloaded model from {REPO}")

DEFAULT_SYSTEM = (
    "You are AegisX, a cybersecurity AI assistant. "
    "Help with recon, scanning, defense, and bug bounty methodology. "
    "Only discuss authorized testing of systems you own or have permission to test.\n\n"
)
SYSTEM_PROMPT = os.environ.get("AEGISX_PROMPT", DEFAULT_SYSTEM)


def respond(message: str, history: list, temperature: float, max_tokens: int) -> str:
    prompt = SYSTEM_PROMPT + "User: " + message + "\n\nAegisX: "
    return generate(
        MODEL_PATH,
        TOKENIZER_PATH,
        prompt,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_k=50,
        repetition_penalty=1.1,
        device="cpu",
    )


with gr.Blocks(title="AegisX-Mini") as demo:
    gr.Markdown("# ⚡ AegisX-Mini\nA lightweight cybersecurity model trained from scratch.")
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Ask about recon, scanning, defense, bug bounty...")
    with gr.Row():
        temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
        max_tokens = gr.Slider(50, 500, value=200, step=50, label="Max new tokens")
    clear = gr.Button("Clear")

    def chat_fn(message, history, temperature, max_tokens):
        reply = respond(message, history, temperature, max_tokens)
        history = history + [(message, reply)]
        return "", history

    msg.submit(chat_fn, [msg, chatbot, temperature, max_tokens], [msg, chatbot])
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()