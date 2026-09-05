"""AegisX-Mini Gradio chat app for Hugging Face Spaces (ZeroGPU-compatible).

Supported layouts:
  1. Model files INSIDE the Space (model.pt + tokenizer.json + optional
     knowledge/ at repo root) -> no env vars needed, simplest possible setup.
  2. Model on the Hub: set AEGISX_REPO env var to your model repo id.

Performance:
  - CPU Spaces: the model is dynamically quantized to int8 at load time
    (2-3x faster inference, ~4x smaller memory footprint).
  - ZeroGPU: fp32 weights move to the allocated GPU (quantization skipped).

RAG grounding (v2):
  If a `knowledge/` folder of .txt files sits next to the app, each question
  first retrieves the top relevant passages (aegisx.rag.CorpusIndex with BM25
  + section-aware chunking), the model answers with that context, and the
  source file names are shown under the reply.

Streaming:
  On CPU the reply streams token-by-token; on ZeroGPU it arrives in one
  piece (the @spaces.GPU decorator runs the generation in a single call).
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

try:
    import spaces
except ImportError:  # pragma: no cover - local runs without the spaces lib
    spaces = None

from aegisx.rag import CorpusIndex

CACHE = Path("/tmp/aegisx-model")
CACHE.mkdir(parents=True, exist_ok=True)

# Layout 1: model files shipped inside the Space repo root.
LOCAL_MODEL = Path(__file__).resolve().parent / "model.pt"
LOCAL_TOKENIZER = Path(__file__).resolve().parent / "tokenizer.json"
LOCAL_KNOWLEDGE = Path(__file__).resolve().parent / "knowledge"

REPO = os.environ.get("AEGISX_REPO", "")
KNOWLEDGE_DIR: Path | None = None
if LOCAL_MODEL.exists() and LOCAL_TOKENIZER.exists():
    MODEL_PATH = str(LOCAL_MODEL)
    TOKENIZER_PATH = str(LOCAL_TOKENIZER)
    print("Using model files shipped inside the Space.")
    if LOCAL_KNOWLEDGE.is_dir() and any(LOCAL_KNOWLEDGE.glob("*.txt")):
        KNOWLEDGE_DIR = LOCAL_KNOWLEDGE
else:
    # Layout 2: fetch from a model repo on the Hub.
    from huggingface_hub import snapshot_download

    if not REPO:
        raise SystemExit("No local model.pt found and AEGISX_REPO env var is not set.")
    snapshot_download(repo_id=REPO, local_dir=str(CACHE))
    MODEL_PATH = str(CACHE / "model.pt")
    TOKENIZER_PATH = str(CACHE / "tokenizer.json")
    if (CACHE / "knowledge").is_dir() and any((CACHE / "knowledge").glob("*.txt")):
        KNOWLEDGE_DIR = CACHE / "knowledge"
    print(f"Downloaded model from {REPO}")

DEFAULT_SYSTEM = (
    "You are AegisX, a cybersecurity AI assistant. "
    "Help with recon, scanning, defense, and bug bounty methodology. "
    "Only discuss authorized testing of systems you own or have permission to test.\n\n"
)
SYSTEM_PROMPT = os.environ.get("AEGISX_PROMPT", DEFAULT_SYSTEM)

# Warm the model once at startup; ZeroGPU reuses it across requests.
_model = None
_index = None
_knowledge_files: list[str] = []


def _quantize_int8(model):
    """Dynamic int8 quantization for the CPU path (Linear layers only)."""
    import torch
    from torch import nn
    from torch.ao.quantization import quantize_dynamic

    if torch.cuda.is_available():
        return model  # GPU path keeps fp32 weights
    try:
        quantized = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        print("Model quantized to int8 (CPU fast path).")
        return quantized
    except Exception as exc:  # quantization is best-effort, never fatal
        print(f"int8 quantization skipped: {exc}")
        return model


def _load_model():
    global _model
    if _model is None:
        from aegisx.model import GPT
        from aegisx.tokenizer import ByteLevelBPETokenizer

        model = GPT.load(MODEL_PATH, device="cpu")
        model.eval()
        model = _quantize_int8(model)
        tokenizer = ByteLevelBPETokenizer.load(TOKENIZER_PATH)
        _model = (model, tokenizer)
        print("Model loaded.")
    return _model


def _load_knowledge() -> tuple[CorpusIndex, list[str]]:
    """Build the retrieval index once over knowledge/*.txt (if any)."""
    global _index, _knowledge_files
    if _index is None and KNOWLEDGE_DIR is not None:
        index = CorpusIndex()
        index.add_dir(KNOWLEDGE_DIR)
        _index = index
        _knowledge_files = sorted({c.source for c in index.chunks})
        print(f"Knowledge base: {len(index)} chunks from {len(_knowledge_files)} files.")
    return _index, _knowledge_files


def _retrieve_context(query: str, top_k: int = 3) -> tuple[str, list[str]]:
    """Return (context_block, source_filenames) for the question, or empty."""
    index, files = _load_knowledge()
    if index is None or not files:
        return "", []
    results = index.retrieve(query, top_k=top_k)
    if not results:
        return "", []
    sources = sorted({c.source for c in results})
    ctx = index.format_context(results, max_chars_per=700)
    return ctx, sources


def _generate_stream(prompt: str, temperature: float, max_tokens: int):
    """Yield progressively longer replies, token by token (CPU path)."""
    import torch

    model, tokenizer = _load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.to(device)
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    prev = idx
    new_token_ids: list[int] = []
    with torch.no_grad():
        for _ in range(int(max_tokens)):
            idx_cond = idx[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(float(temperature), 1e-6)
            for tok in torch.unique(prev[0]):
                val = logits[0, tok]
                logits[0, tok] = val / 1.1 if val > 0 else val * 1.1
            top_vals, top_idx = torch.topk(logits, 50)
            probs = torch.softmax(top_vals, dim=-1)
            nxt = top_idx[0, torch.multinomial(probs, num_samples=1)]
            prev = torch.cat([prev, nxt.view(1, 1)], dim=1)
            idx = prev
            new_token_ids.append(nxt.item())
            text = tokenizer.decode(new_token_ids)
            if "<|endoftext|>" in text:
                break
            yield text.strip()


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


def build_prompt(message: str) -> tuple[str, list[str]]:
    """RAG grounding: pull relevant passages + sources, build the full prompt."""
    context, sources = _retrieve_context(message)
    if context:
        prompt = (
            SYSTEM_PROMPT
            + "Use the references below to answer when they are relevant. "
            + "If the answer is not in the references, say so honestly.\n\n"
            + "References:\n" + context
            + "\n\nUser: " + message + "\n\nAegisX: "
        )
    else:
        prompt = SYSTEM_PROMPT + "User: " + message + "\n\nAegisX: "
    return prompt, sources


def respond(message: str, history: list, temperature: float, max_tokens: int) -> tuple[str, list[str]]:
    """One-shot (non-streaming) reply — kept for API compatibility."""
    prompt, sources = build_prompt(message)
    if spaces is not None:
        reply = _respond_gpu(prompt, temperature, max_tokens)
    else:
        reply = _generate_text(prompt, temperature, max_tokens)
    return reply, sources


def stream_reply(message: str, temperature: float, max_tokens: int):
    """Yield partial replies; one-shot on ZeroGPU, token-by-token on CPU."""
    prompt, _ = build_prompt(message)
    if spaces is not None:
        yield _respond_gpu(prompt, temperature, max_tokens)
    else:
        yield from _generate_stream(prompt, temperature, max_tokens)


if spaces is not None:
    # ZeroGPU: this function runs with a GPU allocated on demand.
    @spaces.GPU
    def _respond_gpu(prompt: str, temperature: float, max_tokens: int) -> str:
        return _generate_text(prompt, temperature, max_tokens)
else:

    def _respond_gpu(prompt: str, temperature: float, max_tokens: int) -> str:
        return _generate_text(prompt, temperature, max_tokens)


with gr.Blocks(title="AegisX-Mini") as demo:
    gr.Markdown(
        "# ⚡ AegisX-Mini\nA lightweight cybersecurity model trained from scratch. "
        "\n\n_Answers are grounded in a local knowledge base when available — "
        "sources appear under each reply. CPU mode streams token-by-token._"
    )
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Ask about recon, scanning, defense, bug bounty...")
    with gr.Row():
        temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
        max_tokens = gr.Slider(50, 500, value=200, step=50, label="Max new tokens")
    clear = gr.Button("Clear")

    def _to_tuples(history):
        """Normalize Chatbot history to [(role, text)] tuples."""
        out = []
        for item in history or []:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content")
                if isinstance(content, list):
                    text = " ".join(
                        c.get("text", "") for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                else:
                    text = str(content or "")
                out.append((role, text))
            else:
                out.append((item[0], item[1]))
        return out

    def _to_messages(history):
        """Convert [(role, text)] tuples to new Gradio message dicts."""
        return [
            {"role": role, "content": [{"text": text, "type": "text"}]}
            for role, text in history
        ]

    def chat_fn(message, history, temperature, max_tokens):
        """Streaming chat handler: yields partials, final yield carries sources."""
        _, sources = build_prompt(message)
        conv = _to_tuples(history) + [("user", message)]
        partial = ""
        for piece in stream_reply(message, temperature, max_tokens):
            partial = piece
            yield "", _to_messages(conv + [("assistant", partial)])
        final = partial.strip()
        if sources:
            final = final + "\n\n📚 Sumber: " + ", ".join(sources)
        yield "", _to_messages(conv + [("assistant", final)])

    msg.submit(chat_fn, [msg, chatbot, temperature, max_tokens], [msg, chatbot])
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
