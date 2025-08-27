# training/train_sft.py
import argparse, os, json, yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import json, os, hashlib, time, torch

def save_metadata(output_dir, meta: dict):
    os.makedirs(output_dir, exist_ok=True)
    meta["timestamp"] = int(time.time())
    s = json.dumps(meta, sort_keys=True)
    meta["hash"] = hashlib.sha256(s.encode()).hexdigest()[:12]
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def device_of():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_config(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--include_kinds", default="positive,refusal",
                    help="comma list of meta.type to include (e.g., positive,refusal)")
    args = ap.parse_args()

    cfg = load_config(args.config)["train"]
    include_kinds = {k.strip() for k in args.include_kinds.split(",") if k.strip()}

    bnb_cfg = None
    if cfg.get("quantization") == "bnb_4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16",
        )

    print("Loading model/tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_cfg
    )

    if not args.no_lora and "lora" in cfg:
        lcfg = cfg["lora"]
        peft_config = LoraConfig(
            r=lcfg["r"],
            lora_alpha=lcfg["lora_alpha"],
            lora_dropout=lcfg["lora_dropout"],
            bias="none",
            target_modules=lcfg["target_modules"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    ds = load_dataset("json", data_files=args.train_jsonl, split="train")

    # Filter: keep only desired meta.type
    def keep(row):
        meta = row.get("meta", {})
        return meta.get("type") in include_kinds
    ds = ds.filter(keep)               

    # Use chat template
    def to_text(row):
        msgs = row["messages"]
        return tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False
        )

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        max_steps=cfg["max_steps"],
        #max_seq_length=cfg["max_seq_length"],
        dataset_text_field="text",
        packing=cfg.get("packing", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        dataset_kwargs={"skip_prepare_dataset": False},
        #report_to=[],
    )

    ds = ds.map(lambda r: {"text": to_text(r)})

    print(f"Training on {len(ds)} rows (kinds={sorted(include_kinds)})...")
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,

        formatting_func=None,

    )

    trainer.train()
    print("Saving...")
    trainer.save_model()
    tok.save_pretrained(args.output_dir)

    save_metadata(args.output_dir, {
        "base_model": args.base_model,
        "config": cfg,
        "train_jsonl": args.train_jsonl,
        "include_kinds": sorted(list(include_kinds)),
    })
    print("Done.")

if __name__ == "__main__":
    main()
