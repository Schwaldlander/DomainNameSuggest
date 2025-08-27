#!/usr/bin/env bash
set -euo pipefail

python data/synthetic/generate_dataset.py \
  --out data/synthetic/sft_dataset.jsonl \
  --n 1200

python training/train_sft.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --train_jsonl data/synthetic/sft_dataset.jsonl \
  --output_dir runs/llama3.2-3b-lora-v1 \
  --config configs/training_lora.yaml

python evaluation/evaluate.py \
  --model runs/llama3.2-3b-lora-v1 \
  --report runs/llama3.2-3b-lora-v1/eval_report.json

python inference/generate.py \
  --model runs/llama3.2-3b-lora-v1 \
  --business "Artisanal coffee shop in Lisbon" \
  --k 12
