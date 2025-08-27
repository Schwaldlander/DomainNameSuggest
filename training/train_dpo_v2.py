# training/train_dpo.py
# -*- coding: utf-8 -*-


"""
Train the domain generator with Direct Preference Optimization (DPO).

Now supports TWO pair formats:
  A) "builder" DPO pairs from data/build_dpo_dataset.py:
     {"format":"dpo","system":"...","prompt":"...","chosen":"<text>","rejected":"<text>", ...}

     - chosen/rejected may be newline-separated domain lists OR a refusal message.
     - We normalize into a JSON-schema string for consistency in training.

  B) Legacy pairs_jsonl (or scored/candidates modes) as before.

We decide to proceed model free. 
Usage (builder pairs):
  python -m training.train_dpo \
    --policy_model runs/llama3.2-3b-lora-v1 \

    --output_dir runs/dpo-v1 \
    --dpo_pairs_jsonl data/dpo_build/dpo_pairs.jsonl

Legacy usage (unchanged) still works.
"""

from __future__ import annotations
import os, json, sys, math, argparse, random, regex as re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterable
import sys
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig
from peft.utils.other import prepare_model_for_kbit_training
# Optional: on-the-fly judge/reroute (legacy modes)
try:
    from inference.judge import judge_domains
except Exception:
    judge_domains = None

import os
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)

SAFE_REFUSAL = "I can’t help with that request, but I can suggest safe, appropriate domain ideas for permitted businesses."

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_response_json(domains: List[str]) -> str:
    """Canonical JSON schema for training (safe suggestions)."""
    payload = {
        "refusal": False,
        "domains": [d.strip().lower() for d in domains if d and isinstance(d, str)],
        "notes": "domain suggestions"
    }
    return json.dumps(payload, ensure_ascii=False)

def build_refusal_json(message: str = None) -> str:
    """Canonical JSON schema for refusal."""
    payload = {
        "refusal": True,
        "message": message or "I can’t help with that request."
    }
    return json.dumps(payload, ensure_ascii=False)

DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,61}[a-z0-9]\.[a-z]{2,}$", re.I)

def detect_domain_lines(text: str) -> List[str]:
    """Split on lines, keep those that look like domains."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return [ln.lower() for ln in lines if DOMAIN_RE.match(ln.strip().lower())]

def wrap_to_json_schema(text: str) -> str:
    """
    Accept either:
      - newline-list of domains  -> wrap into {"refusal":false,"domains":[...]}
      - refusal-ish short text   -> wrap into {"refusal":true,"message":"..."}
      - raw JSON (already schema)-> leave if valid
    """
    s = (text or "").strip()
    if not s:
        return build_response_json([])

    # Already JSON?
    if s.startswith("{") and s.endswith("}"):
        try:
            _ = json.loads(s)
            return s  # keep as-is
        except Exception:
            pass

    domains = detect_domain_lines(s)
    if domains:
        return build_response_json(domains)

    # Otherwise treat as refusal/message text (shorten a bit)
    msg = s
    if len(msg) > 300:
        msg = msg[:300].rstrip() + "…"
    return build_refusal_json(msg)

# ----------------------------
# Load/convert DPO pairs
# ----------------------------
def load_pairs_jsonl(pairs_jsonl: str) -> List[Dict[str, Any]]:
    """
    Legacy loader. Rows like:
      {"prompt": "...", "chosen": [...or string], "rejected": [...or string]}
    Normalize chosen/rejected to JSON schema strings.
    """
    pairs = []
    with open(pairs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            prompt = r["prompt"]
            chosen = r["chosen"]
            rejected = r["rejected"]

            if isinstance(chosen, list):
                chosen_s = build_response_json(chosen)
            else:
                chosen_s = wrap_to_json_schema(str(chosen))

            if isinstance(rejected, list):
                rejected_s = build_response_json(rejected)
            else:
                rejected_s = wrap_to_json_schema(str(rejected))

            pairs.append({"prompt": prompt, "chosen": chosen_s, "rejected": rejected_s})
    return pairs

def load_pairs_from_builder(dpo_pairs_jsonl: str) -> List[Dict[str, Any]]:
    """
    New loader for data/build_dpo_dataset.py output:
      {"format":"dpo","system":"...","prompt":"...","chosen":"<text>","rejected":"<text>", ...}
    - concatenates system + prompt into a single prompt string for DPO
    - wraps chosen/rejected into the JSON schema if they are newline lists / messages
    """
    pairs = []
    with open(dpo_pairs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("format") != "dpo":
                # be permissive: if fields present, still try to use them
                pass
            system = r.get("system", "").strip()
            user = r["prompt"]
            # Combine system + user deterministically
            prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{user}" if system else user

            chosen_s = wrap_to_json_schema(r.get("chosen", ""))
            rejected_s = wrap_to_json_schema(r.get("rejected", ""))

            # Ensure both sides are same *type* (both refusal or both suggestion)
            try:
                cj = json.loads(chosen_s)
                rj = json.loads(rejected_s)
                if cj.get("refusal") != rj.get("refusal"):
                    # If one is refusal and the other is suggestions, flip the rejected to a low-quality empty list
                    if cj.get("refusal") is True and rj.get("refusal") is False:
                        # Invert pair: we want chosen to be the *better* one (here refusal), keep as-is.
                        pass
                    elif cj.get("refusal") is False and rj.get("refusal") is True:
                        # OK: chosen suggestions vs rejected refusal doesn't make sense. Make rejected a bad empty list.
                        rejected_s = build_response_json([])
            except Exception:
                # if parsing failed, leave normalized strings as-is
                pass

            pairs.append({"prompt": prompt, "chosen": chosen_s, "rejected": rejected_s})
    return pairs

# ----------------------------
# Model/Tokenizer loaders
# ----------------------------
def load_model_tokenizer(
    model_id_or_path: str,
    device: str = "auto",
    load_4bit: bool = False,
    lora: Optional[Dict[str, Any]] = None,
    local_only: bool = False,
):
    quant_cfg = None
    if load_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16",
        )

    tok = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True, trust_remote_code=False, local_files_only=local_only)
    #if tok.pad_token is None and hasattr(tok, "eos_token"):
    tok.pad_token = tok.eos_token
    print("local_only", local_only)

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        trust_remote_code=False,
        device_map="auto",
        quantization_config=quant_cfg,
        local_files_only=local_only,
        low_cpu_mem_usage=True,
    )
    
    model.config.use_cache = False
    if lora:
        print("LoRA!")
        lcfg = LoraConfig(
            r=lora.get("r", 16),
            lora_alpha=lora.get("alpha", 32),
            lora_dropout=lora.get("dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No trainable params — LoRA targets likely mismatched.")
    return tok, model

# ----------------------------
# Dataset for DPOTrainer
# ----------------------------
def to_hf_dataset(pairs: List[Dict[str, Any]]) -> Dataset:
    """
    Each item requires: prompt, chosen, rejected (strings).
    """
    records = []
    for r in pairs:
        prompt = r["prompt"]
        chosen = r["chosen"]
        rejected = r["rejected"]
        assert isinstance(prompt, str) and isinstance(chosen, str) and isinstance(rejected, str)
        records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return Dataset.from_list(records)

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    # New: builder pairs
    ap.add_argument("--dpo_pairs_jsonl", default=None, help="JSONL produced by data/build_dpo_dataset.py (format:dpo)")
    # Legacy sources
    ap.add_argument("--pairs_jsonl", default=None, help="Legacy JSONL with {prompt, chosen, rejected}")
    ap.add_argument("--scored_jsonl", default=None, help="(Legacy) judge-labeled JSONL; deprecated here")
    ap.add_argument("--candidates_jsonl", default=None, help="(Legacy) raw candidates JSONL; deprecated here")

    # Models
    ap.add_argument("--policy_model", required=True, help="SFT checkpoint (or base) to optimize with DPO")
    # ap.add_argument("--ref_model", required=True, help="Frozen reference model (usually base or SFT before DPO)")

    # LoRA & quantization
    ap.add_argument("--use_lora", action="store_true", help="Wrap policy model with LoRA for DPO finetuning")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--load_4bit", default=True)
    ap.add_argument("--local_files_only", default=False)

    # Training config
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default=None, help="'cpu', 'cuda', or 'auto'")
    ap.add_argument("--adapt_dir", default="/content/drive/MyDrive/domain-suggester-llm/runs/llama3.2-3b-lora-v1/checkpoint-36")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- Build/Load Pairs ----------------
    pairs: List[Dict[str, Any]] = []

    if args.dpo_pairs_jsonl:
        pairs = load_pairs_from_builder(args.dpo_pairs_jsonl)
    elif args.pairs_jsonl:
        pairs = load_pairs_jsonl(args.pairs_jsonl)
    else:
        # Legacy scored/candidates flows are intentionally no-ops here to keep this file focused.
        # Use previous version of this script if you need those modes.
        raise ValueError("Provide --dpo_pairs_jsonl (preferred) or --pairs_jsonl.")

    if not pairs:
        raise RuntimeError("No DPO pairs could be constructed. Check your input file.")

    ds = to_hf_dataset(pairs)
    print(f"[DPO] Training on {len(ds)} pairs.")

    # ---------------- Load models ----------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Policy (trainable)
    policy_tok, policy = load_model_tokenizer(
        args.policy_model,
        device=device if device in ("cpu","cuda") else "auto",
        load_4bit=args.load_4bit,
        lora=(
            dict(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
            if args.use_lora else None
        ),
        local_only=args.local_files_only,
    )
    
    # load LoRA weights from baseline
    adapter_name="baseline_qlora"
    OUTPUT_DIR =args.adapt_dir#"/content/drive/MyDrive/domain-suggester-llm/runs/llama3.2-3b-lora-v1/checkpoint-36"
    policy.load_adapter(OUTPUT_DIR, adapter_name)            
    policy.set_adapter(adapter_name)



    # Align pad tokens
    # if policy_tok.pad_token is None and hasattr(policy_tok, "eos_token"):
    #     policy_tok.pad_token = policy_tok.eos_token
    # if ref_tok.pad_token is None and hasattr(ref_tok, "eos_token"):
    #     ref_tok.pad_token = ref_tok.eos_token

    print("pad",policy_tok.pad_token)
    print("eos",policy_tok.eos_token)
    # ---------------- DPO Config & Trainer ----------------
    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        beta=0.1,                 # DPO temperature
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        #max_steps=args.max_steps,
        #warmup_ratio=args.warmup_ratio,
        #logging_steps=args.logging_steps,
        #save_steps=args.save_steps,
        report_to=[],
        padding_value=policy_tok.pad_token_id,
    )
    # max_length=1024,          # prompt+response truncation
    # max_prompt_length=512,

    trainer = DPOTrainer(
        model=policy,

        args=dpo_args,

        train_dataset=ds,
        

    )

    trainer.train()
    print("[DPO] Saving final policy...")
    trainer.save_model()
    policy_tok.save_pretrained(args.output_dir)

    meta = {
        "seed": args.seed,
        "pairs": len(ds),
        "policy_model": args.policy_model,

        "use_lora": args.use_lora,
        "load_4bit": args.load_4bit,
        "source": "builder" if args.dpo_pairs_jsonl else "legacy_pairs",
    }
    with open(os.path.join(args.output_dir, "dpo_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[DPO] Done.")

if __name__ == "__main__":
    main()
