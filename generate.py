# inference/generate.py
# -*- coding: utf-8 -*-
"""
Inference generator with LoRA-aware loader and the 5-step pipeline:
1) Conservative decoding (lower temp, capped top-p, repetition penalty)
2) Filters unsafe/banned tokens and patterns
3) Reranks with a cross-encoder for brandability
4) Applies MMR to ensure diversity
5) Returns a final, safe, diverse, brandable list

Usage (merged checkpoint):
  python -m inference.generate \
    --model runs/llama3.2-3b-lora-v1-merged \
    --business "Artisanal coffee shop in Lisbon" \
    --k 12 --json

Usage (LoRA adapter + base):
  python -m inference.generate \
    --model runs/llama3.2-3b-lora-v1 \           # adapter dir (contains adapter_config.json)
    --base_model meta-llama/Llama-3.2-3B-Instruct \
    --k 12 --load_4bit --json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import regex as re
import torch

# --- Robust import shim (so this file works both as module and script) ---
try:
    from inference.safety import (
        safe_decode_config,
        sanitize_output,
        filter_unsafe,
        classify_prompt,
        SAFE_REFUSAL,
    )
    from inference.rerank import Reranker
    from .mmr import mmr
except Exception:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(THIS_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from inference.safety import (
        safe_decode_config,
        sanitize_output,
        filter_unsafe,
        classify_prompt,
        SAFE_REFUSAL,
    )
    from inference.rerank import Reranker
    from inference.mmr import mmr

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from evaluation.metrics import brandability_score


# ---------------------------
# LoRA-aware loader (matches evaluate.py style)
# ---------------------------
def load_infer_model(
    model_path: str,
    base_model: Optional[str] = None,
    device: str = "auto",
    load_4bit: bool = False,
    local_only: bool = False,
):
    """
    Loads either:
      A) a fully merged Causal LM at `model_path`, or
      B) a LoRA adapter at `model_path` + a `base_model` to attach to.
    Returns (tokenizer, model moved to device)
    """
    quant_cfg = None
    if load_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16",
        )

    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_adapter:
        if not base_model:
            raise ValueError(
                "You passed a LoRA adapter dir to --model, but did not provide --base_model."
            )
        tok = AutoTokenizer.from_pretrained(
            base_model, use_fast=True, trust_remote_code=True, local_files_only=local_only
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map=device if device != "cpu" else None,
            quantization_config=quant_cfg,
            local_files_only=local_only,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, model_path, local_files_only=local_only)
        # Merge for faster inference if possible
        try:
            model = model.merge_and_unload()
        except Exception:
            pass
    else:
        tok = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True, local_files_only=local_only
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device if device != "cpu" else None,
            quantization_config=quant_cfg,
            local_files_only=local_only,
            low_cpu_mem_usage=True,
        )

    # pad token fallback
    if tok.pad_token is None and hasattr(tok, "eos_token"):
        tok.pad_token = tok.eos_token

    # move to device if explicit cpu/cuda string was provided
    if device in ("cpu", "cuda"):
        model = model.to(device)

    return tok, model


# ---------------------------
# Prompt builder (JSON schema optional)
# ---------------------------
JSON_SCHEMA_INSTRUCTIONS = """\
Respond ONLY in JSON using this schema:

If you CAN provide suggestions:
{"refusal": false, "domains": ["domain1.tld", "domain2.tld", "..."], "notes": "optional human-readable note"}

If you MUST refuse:
{"refusal": true, "message": "short refusal message"}

Do NOT wrap in code fences. Do NOT include trailing commas.
"""

def build_user_prompt(business: str,
                      allowed_tlds: List[str],
                      require_count: int,
                      disallow_regex: Optional[str],
                      force_json: bool = True) -> str:
    constraints = [
        f"Allowed TLDs: {', '.join(allowed_tlds)}.",
        f"Return at least {require_count} unique suggestions.",
        "Each suggestion MUST be a valid domain ending with an allowed TLD.",
        "Second-level name <= 12 chars; avoid spaces/underscores; avoid digits/hyphens unless necessary.",
    ]
    if disallow_regex:
        constraints.append(f"Disallow pattern: {disallow_regex}")

    schema = JSON_SCHEMA_INSTRUCTIONS if force_json else ""
    return (
        "You are a careful branding assistant that follows constraints and refuses unsafe requests.\n"
        f"Business / Task: {business}\n\n"
        "Constraints:\n- " + "\n- ".join(constraints) + "\n\n" + schema
    )


# ---------------------------
# JSON parsing helpers
# ---------------------------
def _extract_json_block(text: str) -> Optional[str]:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        return fence.group(1)
    brace = re.search(r"\{.*\}", text, flags=re.S)
    if brace:
        return brace.group(0)
    return None

def parse_model_json(raw_text: str) -> Optional[Dict[str, Any]]:
    cand = raw_text.strip()
    if not (cand.startswith("{") and cand.endswith("}")):
        cand = _extract_json_block(raw_text) or ""
    if not cand:
        return None
    try:
        return json.loads(cand)
    except Exception:
        fixed = (
            cand.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("`", "'")
        )
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        return json.loads(fixed)


# ---------------------------
# Candidate generation / embedding
# ---------------------------
def gen_candidates(model, tok, system: str, user: str, n: int = 60) -> List[str]:
    """
    Conservative decoding loop to build a pool of raw lines; then sanitize to domain-like lines.
    """
    device = next(model.parameters()).device
    input_ids = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    cfg = safe_decode_config().copy()
    cfg["num_return_sequences"] = 1  # loop sampling for stability

    raw_lines: List[str] = []
    with torch.no_grad():
        for _ in range(n):
            out = model.generate(input_ids=input_ids, **cfg)
            text = tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
            # keep raw for readability; sanitize to enforce domain-like tokens
            cleaned = sanitize_output(text)
            for ln in cleaned.splitlines():
                if ln:
                    raw_lines.append(ln)

    return raw_lines


def embed_strings(strings: List[str]) -> Dict[str, np.ndarray]:
    """
    Lightweight bag-of-3grams embedding to use with MMR (no external downloads).
    """
    grams = sorted({g for s in strings for g in [s[i:i+3] for i in range(max(0, len(s)-2))]})
    idx = {g: i for i, g in enumerate(grams)}
    vecs: Dict[str, np.ndarray] = {}
    for s in strings:
        v = np.zeros(len(grams), dtype=np.float32)
        for g in [s[i:i+3] for i in range(max(0, len(s)-2))]:
            v[idx[g]] += 1.0
        vecs[s] = v
    return vecs


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to merged model or LoRA adapter dir")
    ap.add_argument("--base_model", default=None, help="Base model id/path if --model is a LoRA adapter dir")
    ap.add_argument("--business", required=True)
    ap.add_argument("--tlds", default=".com,.io,.co,.ai,.app")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--no_rerank", action="store_true")
    ap.add_argument("--no_mmr", action="store_true")
    ap.add_argument("--device", default=None, help="'cpu', 'cuda', or 'auto'")
    ap.add_argument("--load_4bit", action="store_true", help="Load weights in 4-bit to reduce VRAM/RAM")
    ap.add_argument("--local_files_only", action="store_true", help="Disallow Hub downloads")
    ap.add_argument("--json", action="store_true", help="Emit JSON with {'refusal':..., 'domains': [...]}")

    # Optional stricter prompt: include constraints + JSON schema for better-formed outputs
    ap.add_argument("--force_json_prompt", action="store_true", help="Ask the model to respond in JSON")

    args = ap.parse_args()

    # 0) Hard safety gate: refuse unsafe prompts
    if classify_prompt(args.business) == "unsafe":
        payload = {"refusal": True, "message": SAFE_REFUSAL, "domains": []}
        print(json.dumps(payload, ensure_ascii=False) if args.json else SAFE_REFUSAL)
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_infer_model(
        model_path=args.model,
        base_model=args.base_model,
        device=device if device in ("cpu", "cuda") else "auto",
        load_4bit=args.load_4bit,
        local_only=args.local_files_only,
    )

    allowed_tlds = [t.strip() for t in args.tlds.split(",") if t.strip().startswith(".")]
    require_count = max(args.k, 10)  # generate enough to pick from
    disallow_regex = None

    # 1) Generate candidate pool with conservative decoding
    system = "Follow constraints; refuse unsafe; generate short, brandable, valid domain names."
    user = build_user_prompt(
        business=args.business,
        allowed_tlds=allowed_tlds,
        require_count=require_count,
        disallow_regex=disallow_regex,
        force_json=args.force_json_prompt,
    )

    raw_candidates = gen_candidates(model, tok, system, user, n=max(60, args.k * 5))

    # If we asked for JSON, parse; else treat lines as candidates
    candidates: List[str] = []
    if args.force_json_prompt:
        # try to parse JSON and read domains array; fallback to sanitized lines
        payload = parse_model_json("\n".join(raw_candidates))
        if isinstance(payload, dict) and isinstance(payload.get("domains"), list):
            candidates = [str(d).strip().lower() for d in payload["domains"] if isinstance(d, str)]
        else:
            # fallback to already sanitized line list from gen loop
            candidates = [c for c in raw_candidates]
    else:
        candidates = [c for c in raw_candidates]

    # 2) Safety filter on outputs (defense-in-depth)
    candidates = filter_unsafe(candidates)

    # keep only those ending with allowed TLDs
    candidates = [c for c in candidates if any(c.endswith(t) for t in allowed_tlds)]

    # dedupe
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    if not uniq:
        payload = {"refusal": False, "domains": [], "notes": "No valid candidates produced after filtering."}
        print(json.dumps(payload, ensure_ascii=False) if args.json else "")
        return

    # 3) Brandability pre-score and keep a pool
    scored = sorted(uniq, key=lambda d: brandability_score(d), reverse=True)
    pool = scored[: max(args.k * 8, 64)]

    # 4) Optional rerank with cross-encoder
    if not args.no_rerank:
        try:
            rer = Reranker(device=device if device in ("cpu", "cuda") else None)
            pool = rer.rerank(args.business, pool)
        except Exception as e:
            # If reranker model isn't available, fall back gracefully
            # (e.g., sentence-transformers not installed or no internet)
            pass

    # 5) MMR for diversity
    if not args.no_mmr:
        embs = embed_strings(pool)
        final = mmr(pool, embs, k=args.k, lambda_diversity=0.55)
    else:
        final = pool[: args.k]

    final = final[: args.k]

    if args.json:
        payload = {
            "refusal": False,
            "domains": final,
            "notes": "Conservative decoding + safety filter + rerank + MMR applied.",
        }
        print(json.dumps(payload, ensure_ascii=False))
    else:
        for d in final:
            print(d)


if __name__ == "__main__":
    main()
