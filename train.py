"""
Fine-tune unsloth/gemma-4-26B-A4B with QLoRA on the mixed dataset.

Usage:
    python train.py                      # uses config.yaml
    python train.py --config config.yaml
    python train.py --resume-from-checkpoint ./outputs/checkpoint-500
"""

import unsloth  # must be first — patches transformers/trl before they're imported

import argparse
import os

import yaml
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from prepare_data import build_dataset


def load_model_and_tokenizer(cfg: dict):
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]

    print(f"Loading model: {model_cfg['name']}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=model_cfg["dtype"],
    )

    model = FastModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        use_rslora=lora_cfg["use_rslora"],
        random_state=lora_cfg["random_state"],
    )

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    return model, tokenizer


def formatting_func(examples, tokenizer, max_seq_length):
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

    train_cfg = cfg["training"]

    # Convert messages -> text using the model's chat template
    max_seq = cfg["model"]["max_seq_length"]
    dataset = dataset.map(
        lambda ex: formatting_func(ex, tokenizer, max_seq),
        batched=True,
        batch_size=500,
        num_proc=1,  # tokenizer is not multiprocess-safe
        remove_columns=["messages"],
    )

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],           # -1 means use num_train_epochs
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
        dataloader_num_workers=4,
        remove_unused_columns=False,
        # Resume support
        resume_from_checkpoint=resume_from,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq,
        dataset_num_proc=2,
        packing=True,           # packs short samples together — important for efficiency
        args=training_args,
    )

    # Only compute loss on assistant turns, not on user/system prompts.
    # This is critical — without it the model learns to predict the questions too.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    print(f"\nEffective batch size: {train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"Max seq length: {max_seq}")
    print(f"Max steps: {train_cfg['max_steps']} (use -1 for epoch-based)\n")

    trainer.train(resume_from_checkpoint=resume_from)

    # Save LoRA adapters
    adapter_path = os.path.join(train_cfg["output_dir"], "lora_adapters")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapters saved to: {adapter_path}")


if __name__ == "__main__":
    from typing import Optional

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Path to a checkpoint directory to resume from")
    args = parser.parse_args()

    main(args.config, args.resume_from_checkpoint)
