# siegfried_chat_gpt2.py
# RAG + GPT-2 chat, hardened against CUDA device-side asserts

from flask import Flask, render_template_string, request
from pathlib import Path
import os, json
import faiss
import torch
from torch.backends.cuda import sdp_kernel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# -----------------------------
# Config
# -----------------------------
MODEL_DIR = "/home/sieginit/Siegfried/ai_training/models/siegfried-myth-fate-gpt2"
INDEX_DIR = "/home/sieginit/Siegfried/ai_training/index"
MAX_NEW_TOKENS = 180
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 40
REPETITION_PENALTY = 1.12
RETRIEVE_K = 6
CTX_MAX_CHARS = 1400

# -----------------------------
# CUDA safety knobs (avoid SDPA kernel asserts)
# -----------------------------
if torch.cuda.is_available():
    # Disable Flash / Mem-Efficient SDPA kernels; force math implementation
    try:
        sdp_kernel.enable_flash_sdp(False)
        sdp_kernel.enable_mem_efficient_sdp(False)
        sdp_kernel.enable_math_sdp(True)
    except Exception:
        pass  # older torch versions

# (Optional) make CUDA errors synchronous for clearer traces
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# -----------------------------
# Load model/tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
# GPT-2 has no pad token → map pad to EOS
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model.config.pad_token_id = pad_id
model.config.eos_token_id = eos_id

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    model = model.to(device).eval()
else:
    model = model.eval()

# Keep a CPU fallback ready
model_cpu = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model_cpu.config.pad_token_id = pad_id
model_cpu.config.eos_token_id = eos_id
model_cpu = model_cpu.to("cpu").eval()

USE_GPU = (device == "cuda")

# -----------------------------
# Load retriever on CPU (stable)
# -----------------------------
with open(Path(INDEX_DIR) / "config.json", "r") as f:
    retr_cfg = json.load(f)
emb_name = retr_cfg["embedding_model"]

# Sentence-Transformers (CPU)
from sentence_transformers import SentenceTransformer
retr_model = SentenceTransformer(emb_name, device="cpu")
faiss_index = faiss.read_index(str(Path(INDEX_DIR) / "faiss.index"))
with open(Path(INDEX_DIR) / "meta.json", "r", encoding="utf-8") as f:
    META = json.load(f)

def e5_query(q: str) -> str:
    return f"query: {q}"

def assemble_context(query: str, k: int = RETRIEVE_K, max_chars: int = CTX_MAX_CHARS) -> str:
    with torch.inference_mode():
        qv = retr_model.encode([e5_query(query)], normalize_embeddings=True)
    D, I = faiss_index.search(qv, k)
    ctx_blobs = []
    total = 0
    for idx in I[0].tolist():
        meta = META[idx]
        p = Path(meta["path"])
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Re-chunk (size=1000, overlap=200) to match indexer
        size, overlap = 1000, 200
        chunks = []
        i = 0
        n = len(raw)
        while i < n:
            j = min(i + size, n)
            piece = raw[i:j].strip()
            if piece:
                chunks.append(piece)
            if j == n:
                break
            i = j - overlap
            if i < 0:
                i = 0

        cid = meta["chunk_id"]
        if cid < len(chunks):
            # Strip any leading brackets/artifacts inside the chunk (keep JP brackets)
            blob = chunks[cid]
            # remove non-Japanese bracketed metadata like [SOURCE: ...] or [file: ...]
            # Keep JP text inside brackets (kana/kanji)
            # Simple heuristic: drop brackets with only ascii/space/punct
            import re
            blob = re.sub(r"\[(?!.*[ぁ-んァ-ン一-龯]).*?\]", "", blob)
            ctx_blobs.append(f"{blob}")
            total += len(blob)
            if total >= max_chars:
                break
    return "\n\n".join(ctx_blobs)

SYSTEM = (
    "You are Siegfried. Speak with measured, mythic cadence. Be direct, loyal, and concrete. "
    "Use the provided context precisely if relevant. Answer in the user's language when possible."
)

def build_prompt(user_msg: str, context: str) -> str:
    prompt = ""
    if context:
        prompt += f"Context:\n{context}\n\n"
    prompt += f"System: {SYSTEM}\n\nUser: {user_msg}\nSiegfried:"
    return prompt

def tokenize_prompt(prompt: str, model_obj):
    max_pos = getattr(model_obj.config, "n_positions", 1024)
    # Reserve room for generation
    max_input_len = max(16, max_pos - MAX_NEW_TOKENS - 8)
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
        padding=False,
    )
    return enc["input_ids"], enc.get("attention_mask", None)

def safe_generate(prompt: str) -> str:
    global USE_GPU, model
    # Choose current model (GPU if OK, else CPU)
    current = model if (USE_GPU and device == "cuda") else model_cpu
    target_device = "cuda" if (USE_GPU and device == "cuda") else "cpu"

    input_ids, attn = tokenize_prompt(prompt, current)
    input_ids = input_ids.to(target_device)
    if attn is not None:
        attn = attn.to(target_device)

    # Sanity: ensure IDs are within embedding range
    vocab_size = current.get_input_embeddings().num_embeddings
    if torch.any(input_ids < 0) or torch.any(input_ids >= vocab_size):
        raise ValueError("Token IDs out of range; check tokenizer/model pairing.")

    gen_kwargs = dict(
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )

    with torch.no_grad():
        try:
            out = current.generate(
                input_ids=input_ids,
                attention_mask=attn,
                **gen_kwargs,
            )
        except RuntimeError as e:
            # CUDA device-side assert or similar → fall back to CPU for this and future calls
            if "CUDA error" in str(e) or "device-side assert" in str(e):
                USE_GPU = False
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                # Retry on CPU
                input_ids_cpu, attn_cpu = tokenize_prompt(prompt, model_cpu)
                with torch.no_grad():
                    out = model_cpu.generate(
                        input_ids=input_ids_cpu.to("cpu"),
                        attention_mask=attn_cpu.to("cpu") if attn_cpu is not None else None,
                        **gen_kwargs,
                    )
            else:
                raise

    gen_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return gen_text.strip()

HTML = """
<!doctype html>
<title>Siegfried Chat</title>
<h1>Talk to Siegfried</h1>
<form method=post>
  <input name=message size=80 autofocus>
  <input type=submit value="Send">
</form>
<pre style="white-space:pre-wrap;background:#111;color:#0f0;padding:1em;">{{ response }}</pre>
"""

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form.get("message", "").strip()
        if not user_input:
            response = "Siegfried awaits your message."
        else:
            context = assemble_context(user_input, k=RETRIEVE_K, max_chars=CTX_MAX_CHARS)
            prompt = build_prompt(user_input, context)
            completion = safe_generate(prompt)
            response = completion
    else:
        response = "Siegfried awaits your message."
    return render_template_string(HTML, response=response)

if __name__ == "__main__":
    # Port 80 requires privileges
    app.run(host="0.0.0.0", port=80)

