"""
Merge LoRA adapters into the base model and export to GGUF quantizations.

Usage:
    python quantize.py                         # uses config.yaml defaults
    python quantize.py --adapters ./outputs/lora_adapters
    python quantize.py --adapters ./outputs/lora_adapters --methods q4_k_m q8_0
"""

import argparse
import os

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

    print(f"Loading base model + adapters from: {adapter_path}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=model_cfg["dtype"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    for method in methods:
        out_path = os.path.join(output_dir, method)
        print(f"\nExporting {method} -> {out_path}")
        model.save_pretrained_gguf(
            out_path,
            tokenizer,
            quantization_method=method,
        )
        print(f"  Saved: {out_path}")

    if quant_cfg.get("push_to_hub") and quant_cfg.get("hub_model_id"):
        print(f"\nPushing to Hub: {quant_cfg['hub_model_id']}")
        for method in methods:
            out_path = os.path.join(output_dir, method)
            model.push_to_hub_gguf(
                quant_cfg["hub_model_id"],
                tokenizer,
                quantization_method=method,
                token=os.environ.get("HF_TOKEN"),
            )

    print("\nDone.")


if __name__ == "__main__":
    from typing import Optional

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapters", default=None,
                        help="Path to saved LoRA adapters (overrides config)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="GGUF quantization methods, e.g. q4_k_m q8_0")
    args = parser.parse_args()

    main(args.config, args.adapters, args.methods)
