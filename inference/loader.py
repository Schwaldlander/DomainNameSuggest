'''
Helper script for efficient model loading
'''
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
TRUST_REMOTE_CODE = False

def load_eval_model(
    model_path: str,
    base_model: str | None = None,
    device: str = "auto",
    load_4bit: bool = False,
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

    # Detect if model_path is a PEFT adapter by checking for adapter config
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_adapter:
        if not base_model:
            raise ValueError(
                "You passed a LoRA adapter dir to --model. Please also pass --base_model "
                "so we can load base weights and attach the adapter."
            )
        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=TRUST_REMOTE_CODE, local_files_only=local_only)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=TRUST_REMOTE_CODE,
            device_map=device if device != "cpu" else None,
            quantization_config=quant_cfg,
            local_files_only=local_only,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, model_path, local_files_only=local_only)
        # optionally merge to speed up eval and reduce adapter indirection
        try:
            model = model.merge_and_unload()
        except Exception:
            pass
        return tok, model
    else:
        # Fully merged checkpoint
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=TRUST_REMOTE_CODE, local_files_only=local_only)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=TRUST_REMOTE_CODE,
            device_map=device if device != "cpu" else None,
            quantization_config=quant_cfg,
            local_files_only=local_only,
            low_cpu_mem_usage=True,
        )
        return tok, model
