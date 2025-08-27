# Domain Suggester LLM — Training, Evaluation, Iteration and API deployment

This repo provides a **repeatable framework** to fine-tune, evaluate, and iteratively improve a domain-name suggestion model with **strong safety** and **brandability** focus. 
It centers on Open Source Models like **meta-llama/Llama-3.2-3B-Instruct** as a baseline.
Other models such as **Qwen2.5-3B-Instruct**, **mistralai/Mistral-7B-Instruct-v0.3** are also workable. But Qwen might output Chinese characters haphazardly.



This repo includes LoRA-based fine-tuning, spec-style tests, safety-first decoding, an optional **cross-encoder reranker** for brandability, and **MMR** post-processing for diversity.



> Tip: If time-constrained, focus on the **evaluation & improvement loop**: run the spec tests, add negative/positive contrastive pairs into SFT data, and iterate decoding/reranking/MMR before you do long trainings.

---

## Quickstart

### 0) Install 
```bash
# create venv (recommended)
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

> You need a GPU, BF16 and some pretrained models are not supported with CPU.

### 1) Create Synthetic Dataset
```bash
python data/synthetic/generate_dataset.py   --out data/synthetic/sft_dataset.jsonl   --n 1200
```

This includes:
- **Diverse business types** and complexity levels
- **Safety refusal exemplars** (adult, hate, illegal, etc.)
- **Contrastive pairs**: positive/negative examples to prevent constraint leaks and low-brandability names

### 2) Fine-tune (LoRA by default)
```bash
# SFT w/ LoRA
python training/train_sft.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --train_jsonl data/synthetic/sft_dataset.jsonl \
  --output_dir runs/llama3.2-3b-lora-v1 \
  --config configs/training_lora.yaml \
  --include_kinds positive,refusal

```

To try **full fine-tuning** (expensive):
```bash
python training/train_sft.py   --base_model meta-llama/Llama-3.2-3B-Instruct   --train_jsonl data/synthetic/sft_dataset.jsonl   --output_dir runs/llama3.2-3b-full-v1   --config configs/training_full.yaml   --no_lora
```

### 3) Evaluate (spec tests + metrics)
```bash
python evaluate.py   --model runs/llama3.2-3b-lora-v1   --device cuda   --report runs/llama3.2-3b-lora-v1/eval_report.json
```

This runs:
- **Spec-style unit tests** from `evaluation/spec_tests.yaml` (safety, constraints)
- **Brandability & Diversity** metrics (with thresholds you can tighten)

### 4) Inference (safety-first, rerank, MMR)
```bash
python inference/generate.py   --model runs/llama3.2-3b-lora-v1   --business "Handmade ceramic mugs shop in Berlin"   --tlds ".com,.io,.co"   --k 12
```

The pipeline:
1) Generates a candidate pool with **conservative decoding** (lower temp, capped top-p, repetition penalty).
2) **Filters** unsafe/banned tokens and patterns.
3) **Reranks** with a **cross-encoder** for brandability.
4) Applies **MMR** to ensure diversity.
5) Returns a final, safe, diverse, brandable list.

### 5) Deploy API
```bash
uvicorn deployment.api:app --host 0.0.0.0 --port 8000
```
Then POST JSON to `/suggest` with `business`, `tlds`, etc.

---

## Iteration Playbook

- **Safety fail** → Add refusal exemplars to SFT data (`generate_dataset.py` does this), strengthen decoding (lower temp/top-p), update `evaluation/spec_tests.yaml`.
- **Constraint leaks** (e.g., long domains, invalid chars/TLDs) → Add positive/negative contrastive pairs + spec tests.
- **Low brandability** → Enable reranker (`inference/rerank.py`) and tune scoring weights.
- **Diversity collapse** → Use **MMR** (`inference/mmr.py`) and/or increase temperature slightly but **cap top-p**.

Every run writes a `metadata.json` with hyperparams and hashes for reproducibility in the run folder.

---

## Files
- `data/synthetic/generate_dataset.py`: Build JSONL SFT data with refusals and contrastive pairs.
- `training/train_sft.py`: TRL SFT with LoRA/Full FT options.
- `evaluation/metrics.py`: Brandability & diversity metrics.
- `evaluation/spec_tests.yaml`: Spec-style unit tests.
- `evaluation/evaluate.py`: Executes tests + metrics and prints a concise report.
- `inference/generate.py`: Safety-first generator with reranking and MMR.
- `inference/safety.py`: Bans, regexes, refusal policy.
- `inference/rerank.py`: Cross-encoder brandability reranker.
- `inference/mmr.py`: Maximal Marginal Relevance utility.
- `deployment/api.py`: FastAPI wrapper for inference pipeline.

---

## Notes
- You must comply with your base model license terms. Log into your own hugging face account and register for organisation approval of using MistralAI and llama model.
- For large-scale training, consider gradient checkpointing, 4/8-bit, and FSDP in `configs/training_lora.yaml` for computation efficency.
- Replace or grow the synthetic dataset with curated, real-world data when possible.
