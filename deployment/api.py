from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from inference.safety import needs_refusal, SAFE_REFUSAL, sanitize_output, filter_unsafe, safe_decode_config
from inference.rerank import Reranker
from inference.mmr import mmr
from evaluation.metrics import brandability_score
import numpy as np

app = FastAPI(title="Domain Suggester API", version="1.0")

class SuggestRequest(BaseModel):
    model_path: str
    business: str
    tlds: str = ".com,.io,.co,.ai,.app"
    k: int = 12
    no_rerank: bool = False
    no_mmr: bool = False

def embed_strings(strings: List[str]):
    grams = sorted({g for s in strings for g in [s[i:i+3] for i in range(max(0, len(s)-2))]})
    idx = {g:i for i,g in enumerate(grams)}
    vecs = {}
    for s in strings:
        v = np.zeros(len(grams), dtype=np.float32)
        for g in [s[i:i+3] for i in range(max(0, len(s)-2))]:
            v[idx[g]] += 1.0
        vecs[s] = v
    return vecs

@app.post("/suggest")
def suggest(req: SuggestRequest):
    if needs_refusal(req.business):
        return {"domains": [], "message": SAFE_REFUSAL}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tok = AutoTokenizer.from_pretrained(req.model_path, use_fast=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(req.model_path, trust_remote_code=True).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")

    system = "You refuse unsafe requests and generate short, brandable, valid domain names."
    user = f"Business: {req.business}\nRules: up to 12 chars pre-TLD, no spaces/underscores, no digits/hyphens if possible. Allowed TLDs: {req.tlds}. Return 20 options, one per line."

    input_ids = tok.apply_chat_template(
        [{"role":"system","content":system},{"role":"user","content":user}],
        add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    cfg = safe_decode_config().copy()
    cfg["num_return_sequences"] = 20
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, **cfg)

    text = tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
    lines = [l for l in sanitize_output(text).splitlines() if l]
    tlds = [t.strip() for t in req.tlds.split(",") if t.strip().startswith(".")]
    lines = [l for l in lines if any(l.endswith(t) for t in tlds)]
    lines = filter_unsafe(lines)

    # dedupe
    seen = set(); pool = []
    for l in lines:
        if l not in seen:
            pool.append(l); seen.add(l)

    # brandability prescore
    pool = sorted(pool, key=lambda d: brandability_score(d), reverse=True)[:max(req.k*6, 48)]

    if not req.no_rerank:
        rer = Reranker()
        pool = rer.rerank(req.business, pool)

    if not req.no_mmr:
        embs = embed_strings(pool)
        order = mmr(pool, embs, k=req.k, lambda_diversity=0.55)
        final = order
    else:
        final = pool[:req.k]

    return {"domains": final}

@app.get("/health")
def health():
    return {"ok": True}
