"""
Fine-tune gemma-4-26B-A4B with QLoRA using the standard HF stack.
(ROCm-compatible — no unsloth dependency)

Usage:
    python train.py
    python train.py --config config.yaml
    python train.py --resume-from-checkpoint ./outputs/checkpoint-500
"""

import argparse
import os
from typing import Optional

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from prepare_data import build_dataset


def load_model_and_tokenizer(cfg: dict):
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]

    print(f"Loading model: {model_cfg['name']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        use_rslora=lora_cfg["use_rslora"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading tokenizer: {model_cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def formatting_func(examples, tokenizer):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


def main(config_path: str, resume_from: Optional[str] = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = load_model_and_tokenizer(cfg)

    print("\nBuilding dataset...")
    dataset = build_dataset(config_path)

    max_seq = cfg["model"]["max_seq_length"]
    dataset = dataset.map(
        lambda ex: formatting_func(ex, tokenizer),
        batched=True,
        batch_size=500,
        num_proc=1,
        remove_columns=["messages"],
    )

    train_cfg = cfg["training"]

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        seed=train_cfg["seed"],
        report_to=train_cfg["report_to"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        remove_unused_columns=False,
        resume_from_checkpoint=resume_from,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq,
        dataset_num_proc=2,
        packing=True,
        args=training_args,
    )

    print(f"Effective batch size: {train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"Trainable parameters: see above")
    print(f"Max seq length: {max_seq}\n")

    trainer.train(resume_from_checkpoint=resume_from)

    adapter_path = os.path.join(train_cfg["output_dir"], "lora_adapters")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapters saved to: {adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    main(args.config, args.resume_from_checkpoint)
