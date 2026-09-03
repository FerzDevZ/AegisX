# Deploy AegisX-Mini on Hugging Face (free)

Two parts: **Model Hub** (weights) + **Space** (chat UI).

## 1. Push the checkpoint to the Hub

```bash
pip install -r requirements.txt
HF_TOKEN=hf_xxx python hf/push_to_hub.py \
    --repo youruser/aegisx-mini \
    --checkpoint checkpoints/aegisx-mini
```

(Or push straight from the Colab notebook — cell 6 does exactly this.)

## 2. Create the Space

1. Go to https://huggingface.co/new-space
2. **SDK:** Gradio · **Hardware:** CPU basic (free) · **Space name:** `aegisx-mini`
3. Upload these files to the Space (or point it at this repo's `hf/` folder):
   - `space_app.py`
   - `requirements.txt`
   - `aegisx/` (the Python package — the app imports `aegisx.chat`)
4. In the Space settings, add a **Secret**:
   - `AEGISX_REPO` = `youruser/aegisx-mini`
5. The Space builds and shows a public URL: `https://youruser-aegisx-mini.hf.space`

## 3. Try it

Open the Space URL and chat with AegisX. Since the model is 15–200 MB, the
free CPU tier responds in near-real-time.

> Note: `space_app.py` imports `aegisx/` — upload the package folder alongside
> it. The `hf/` folder here is a template; either duplicate `aegisx/` into the
> Space repo or make the Space a copy of the whole project repo and delete the
> training files you don't need.