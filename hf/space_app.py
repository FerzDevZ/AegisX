"""AegisX-Mini Gradio chat app for Hugging Face Spaces (ZeroGPU-compatible).

Supported layouts:
  1. Model files INSIDE the Space (model.pt + tokenizer.json at repo root)
     -> no env vars needed, simplest possible setup.
  2. Model on the Hub: set AEGISX_REPO env var to your model repo id.

The app is decorated with @spaces.GPU so it runs on the free ZeroGPU tier.
Model inference falls back to CPU when no GPU is allocated.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

try:
    import spaces
except ImportError:  # pragma: no cover - local runs without the spaces lib
    spaces = None

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

# Warm the model once at startup; ZeroGPU reuses it across requests.
_model = None


def _load_model():
    global _model
    if _model is None:
        from aegisx.model import GPT
        from aegisx.tokenizer import ByteLevelBPETokenizer

        model = GPT.load(MODEL_PATH, device="cpu")
        model.eval()
        tokenizer = ByteLevelBPETokenizer.load(TOKENIZER_PATH)
        _model = (model, tokenizer)
        print("Model loaded.")
    return _model


def _generate_text(prompt: str, temperature: float, max_tokens: int) -> str:
    import torch

    model, tokenizer = _load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.to(device)  # ZeroGPU: move weights to the allocated GPU
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_k=50,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(out[0].tolist()[len(ids):]).strip()


def respond(message: str, history: list, temperature: float, max_tokens: int) -> str:
    prompt = SYSTEM_PROMPT + "User: " + message + "\n\nAegisX: "
    if spaces is not None:
        return _respond_gpu(prompt, temperature, max_tokens)
    return _generate_text(prompt, temperature, max_tokens)


if spaces is not None:
    # ZeroGPU: this function runs with a GPU allocated on demand.
    @spaces.GPU
    def _respond_gpu(prompt: str, temperature: float, max_tokens: int) -> str:
        return _generate_text(prompt, temperature, max_tokens)
else:

    def _respond_gpu(prompt: str, temperature: float, max_tokens: int) -> str:
        return _generate_text(prompt, temperature, max_tokens)


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