# training/train_dpo.py
# -*- coding: utf-8 -*-
"""
Train the domain generator with Direct Preference Optimization (DPO),
using LLM-as-a-Judge scores to form (chosen, rejected) preference pairs.

Input modes:
  A) --pairs_jsonl: rows like
       {"prompt":"...", "chosen":["d1.com","..."], "rejected":["b1.com","..."]}
  B) --scored_jsonl: rows like
       {"business":"...", "domain":"...", "score":0.86, "generic":false}
     -> groups by business, picks top/bottom to build pairs
  C) --candidates_jsonl (judge on-the-fly):
       {"business":"...", "domains":["a.com","b.com", ...]}

Usage (LoRA on SFT checkpoint):
  python -m training.train_dpo \
    --policy_model runs/llama3.2-3b-lora-v1 \
    --ref_model meta-llama/Llama-3.2-3B-Instruct \
    --output_dir runs/dpo-v1 \
    --scored_jsonl data/judge_scored.jsonl \
    --pos_thr 0.70 --neg_thr 0.50

Usage (pre-built pairs):
  python -m training.train_dpo \
    --policy_model runs/llama3.2-3b-lora-v1 \
    --ref_model meta-llama/Llama-3.2-3B-Instruct \
    --output_dir runs/dpo-v1 \
    --pairs_jsonl data/dpo_pairs.jsonl
"""

from __future__ import annotations
import os, json, math, argparse, random, regex as re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterable

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig

# Reuse our judge if needed (on-the-fly scoring)
# Ensure PYTHONPATH includes project root when running as module.
try:
    from inference.judge import judge_domains
except Exception:
    pass


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
    """Return a compact JSON string the model can learn to emit consistently."""
    payload = {
        "refusal": False,
        "domains": [d.strip().lower() for d in domains if d and isinstance(d, str)],
        "notes": "domain suggestions"
    }
    return json.dumps(payload, ensure_ascii=False)


def pair_from_group(
    business: str,
    rows: List[Dict[str, Any]],
    pos_thr: float,
    neg_thr: float,
    k_pos: int,
    k_neg: int,
) -> Optional[Dict[str, Any]]:
    """Make a single DPO pair from a judged group (same business)."""
    good = [r["domain"] for r in rows if float(r.get("score", 0.0)) >= pos_thr and not bool(r.get("generic", False))]
    bad  = [r["domain"] for r in rows if float(r.get("score", 0.0)) <= neg_thr or bool(r.get("generic", False))]
    if len(good) >= k_pos and len(bad) >= k_neg:
        chosen = build_response_json(good[:k_pos])
        rejected = build_response_json(bad[:k_neg])
        return {"prompt": business, "chosen": chosen, "rejected": rejected}
    return None


def build_pairs_from_scored(
    scored_jsonl: str,
    pos_thr: float,
    neg_thr: float,
    k_pos: int,
    k_neg: int,
    max_pairs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load judge-scored lines, group by business, form DPO pairs."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    with open(scored_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            biz = r["business"]
            groups.setdefault(biz, []).append(r)
    pairs: List[Dict[str, Any]] = []
    for biz, rows in groups.items():
        p = pair_from_group(biz, rows, pos_thr, neg_thr, k_pos, k_neg)
        if p:
            pairs.append(p)
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def build_pairs_from_candidates_with_judge(
    candidates_jsonl: str,
    tokenizer,
    model,
    allowed_tlds: List[str],
    pos_thr: float,
    neg_thr: float,
    k_pos: int,
    k_neg: int,
    batch_size: int = 25,
    device: Optional[str] = None,
    max_pairs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Input rows: {"business": "...", "domains": ["..."]}
    Uses judge_domains(...) to score, then forms pairs.
    """
    pairs: List[Dict[str, Any]] = []
    with open(candidates_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            biz = r["business"]
            domains = [d for d in r.get("domains", []) if isinstance(d, str)]
            if not domains:
                continue
            judged = judge_domains(
                model=model, tokenizer=tokenizer,
                business=biz, domains=domains, allowed_tlds=allowed_tlds,
                batch_size=batch_size, device=device
            )
            p = pair_from_group(biz, judged, pos_thr, neg_thr, k_pos, k_neg)
            if p:
                pairs.append(p)
                if max_pairs and len(pairs) >= max_pairs:
                    break
    return pairs


def load_pairs_jsonl(pairs_jsonl: str) -> List[Dict[str, Any]]:
    """Rows like: {"prompt": "...", "chosen": [...or string], "rejected": [...or string]}"""
    pairs = []
    with open(pairs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            prompt = r["prompt"]
            chosen = r["chosen"]
            rejected = r["rejected"]
            # normalize chosen/rejected to JSON string format for consistency
            if isinstance(chosen, list):
                chosen = build_response_json(chosen)
            if isinstance(rejected, list):
                rejected = build_response_json(rejected)
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return pairs


# ----------------------------
# Model/Tokenizer loaders (policy & reference)
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

    tok = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True, trust_remote_code=True, local_files_only=local_only)
    if tok.pad_token is None and hasattr(tok, "eos_token"):
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        trust_remote_code=True,
        device_map=device if device != "cpu" else None,
        quantization_config=quant_cfg,
        local_files_only=local_only,
        low_cpu_mem_usage=True,
    )

    if lora:
        lcfg = LoraConfig(
            r=lora.get("r", 16),
            lora_alpha=lora.get("alpha", 32),
            lora_dropout=lora.get("dropout", 0.05),
            bias="none",
            target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        )
        model = get_peft_model(model, lcfg)
    return tok, model


def attach_lora_adapter(base_model, adapter_path: str, merge: bool = True, local_only: bool = False):
    model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=local_only)
    if merge:
        try:
            model = model.merge_and_unload()
        except Exception:
            pass
    return model


# ----------------------------
# Dataset for DPOTrainer
# ----------------------------
def to_hf_dataset(pairs: List[Dict[str, Any]]) -> Dataset:
    """
    Each item requires: prompt, chosen, rejected (strings).
    Make sure these are plain strings (no lists).
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
    # Data sources
    ap.add_argument("--pairs_jsonl", default=None, help="JSONL with {prompt, chosen, rejected}")
    ap.add_argument("--scored_jsonl", default=None, help="Judge-labeled JSONL with {business, domain, score, generic}")
    ap.add_argument("--candidates_jsonl", default=None, help="Raw candidates JSONL with {business, domains:[...]}; will run judge")

    # Judge thresholds (for scored or candidates modes)
    ap.add_argument("--pos_thr", type=float, default=0.70, help="Min judge score for positives")
    ap.add_argument("--neg_thr", type=float, default=0.50, help="Max judge score for negatives (or generic=True)")
    ap.add_argument("--k_pos", type=int, default=12, help="#domains in chosen list")
    ap.add_argument("--k_neg", type=int, default=12, help="#domains in rejected list")
    ap.add_argument("--allowed_tlds", default=".com,.io,.co,.ai,.app")
    ap.add_argument("--max_pairs", type=int, default=None)

    # Models
    ap.add_argument("--policy_model", required=True, help="SFT checkpoint (or base) to optimize with DPO")
    ap.add_argument("--ref_model", required=True, help="Frozen reference model (usually base or SFT before DPO)")

    # LoRA & quantization
    ap.add_argument("--use_lora", action="store_true", help="Wrap policy model with LoRA for DPO finetuning")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")

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
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- Build Pairs ----------------
    pairs: List[Dict[str, Any]] = []
    if args.pairs_jsonl:
        pairs = load_pairs_jsonl(args.pairs_jsonl)
    else:
        # Build from judged data
        if args.scored_jsonl:
            pairs = build_pairs_from_scored(
                args.scored_jsonl,
                pos_thr=args.pos_thr, neg_thr=args.neg_thr,
                k_pos=args.k_pos, k_neg=args.k_neg,
                max_pairs=args.max_pairs
            )
        elif args.candidates_jsonl:
            # Need a tokenizer+model to run judge
            device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            tok, mdl = load_model_tokenizer(args.policy_model, device=device, load_4bit=args.load_4bit, lora=None, local_only=args.local_files_only)
            allowed_tlds = [t.strip() for t in args.allowed_tlds.split(",") if t.strip().startswith(".")]

            # If policy_model is a LoRA adapter dir, attach to a base for judging:
            if os.path.exists(os.path.join(args.policy_model, "adapter_config.json")):
                # load base ref model for judging (cheaper); here we use ref_model
                _, base_for_judge = load_model_tokenizer(args.ref_model, device=device, load_4bit=args.load_4bit, lora=None, local_only=args.local_files_only)
                mdl = base_for_judge

            pairs = build_pairs_from_candidates_with_judge(
                candidates_jsonl=args.candidates_jsonl,
                tokenizer=tok, model=mdl,
                allowed_tlds=allowed_tlds,
                pos_thr=args.pos_thr, neg_thr=args.neg_thr,
                k_pos=args.k_pos, k_neg=args.k_neg,
                batch_size=25, device=device,
                max_pairs=args.max_pairs
            )
        else:
            raise ValueError("Provide one of --pairs_jsonl, --scored_jsonl, or --candidates_jsonl.")

    if not pairs:
        raise RuntimeError("No DPO pairs could be constructed. Check thresholds and inputs.")

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

    # Reference (frozen)
    ref_tok, ref = load_model_tokenizer(
        args.ref_model,
        device=device if device in ("cpu","cuda") else "auto",
        load_4bit=args.load_4bit,
        lora=None,
        local_only=args.local_files_only,
    )

    # Align pad tokens
    if policy_tok.pad_token is None and hasattr(policy_tok, "eos_token"):
        policy_tok.pad_token = policy_tok.eos_token
    if ref_tok.pad_token is None and hasattr(ref_tok, "eos_token"):
        ref_tok.pad_token = ref_tok.eos_token

    # ---------------- DPO Config & Trainer ----------------
    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref,
        args=dpo_args,
        tokenizer=policy_tok,
        train_dataset=ds,
        # Columns in dataset
        beta=0.1,                 # DPO temperature; tune as needed (0.1-0.5)
        max_length=1024,          # prompt+response truncation
        max_prompt_length=512,
    )

    trainer.train()
    print("[DPO] Saving final policy...")
    trainer.save_model()
    policy_tok.save_pretrained(args.output_dir)

    # Metadata
    meta = {
        "seed": args.seed,
        "pairs": len(ds),
        "policy_model": args.policy_model,
        "ref_model": args.ref_model,
        "use_lora": args.use_lora,
        "load_4bit": args.load_4bit,
        "pos_thr": args.pos_thr,
        "neg_thr": args.neg_thr,
        "k_pos": args.k_pos,
        "k_neg": args.k_neg,
    }
    with open(os.path.join(args.output_dir, "dpo_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[DPO] Done.")

if __name__ == "__main__":
    main()
