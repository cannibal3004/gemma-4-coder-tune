"""
Merge LoRA adapters into the base model and export to GGUF quantizations,
then optionally upload to HuggingFace Hub.

Usage:
    python quantize.py
    python quantize.py --adapters /workspace/outputs/lora_adapters
    python quantize.py --adapters /workspace/outputs/lora_adapters --methods q4_k_m q8_0

HF upload:
    Set push_to_hub: true and hub_model_id: "yourname/model-name" in config.yaml
    Export HF_TOKEN env var with a write-access token before running.
"""

import argparse
import os
import sys
from typing import Optional

import torch
import yaml
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template


def main(config_path: str, adapter_path: Optional[str], methods: Optional[list[str]]):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    quant_cfg = cfg["quantize"]

    adapter_path = adapter_path or os.path.join(cfg["training"]["output_dir"], "lora_adapters")
    output_dir = quant_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    methods = methods or quant_cfg["methods"]

    push = quant_cfg.get("push_to_hub", False)
    hub_model_id = quant_cfg.get("hub_model_id", "")
    hf_token = os.environ.get("HF_TOKEN")

    if push and not hub_model_id:
        print("ERROR: push_to_hub is true but hub_model_id is not set in config.yaml")
        sys.exit(1)
    if push and not hf_token:
        print("ERROR: push_to_hub is true but HF_TOKEN env var is not set")
        print("  Get a token at https://huggingface.co/settings/tokens (write access)")
        print("  Then: export HF_TOKEN=hf_...")
        sys.exit(1)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(model_cfg["dtype"], torch.bfloat16)

    print(f"Loading base model + adapters from: {adapter_path}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=dtype,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    for method in methods:
        out_path = os.path.join(output_dir, method)
        print(f"\nExporting {method} -> {out_path}")
        model.save_pretrained_gguf(out_path, tokenizer, quantization_method=method)
        print(f"  Saved: {out_path}")

        if push:
            print(f"  Uploading {method} to {hub_model_id}...")
            model.push_to_hub_gguf(
                hub_model_id,
                tokenizer,
                quantization_method=method,
                token=hf_token,
            )
            print(f"  Uploaded: https://huggingface.co/{hub_model_id}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapters", default=None,
                        help="Path to saved LoRA adapters (overrides config)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="GGUF quantization methods, e.g. q4_k_m q8_0")
    args = parser.parse_args()

    main(args.config, args.adapters, args.methods)
