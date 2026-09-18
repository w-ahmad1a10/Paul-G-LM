#!/usr/bin/env python3
"""
Stage 2 ONLY: Supervised Fine-Tuning directly on base model (no CPT).

Fine-tunes LiquidAI/LFM2.5-230M directly on Q&A data using SFT.
This produces a model that has ONLY SFT training (no continued pre-training).
Used as a comparison model in head-to-head evaluation.

Runs on Colab T4 GPU.

Usage (Colab):
    colab run --gpu T4 python src/training/train_stage2_sftonly.py

Or locally if GPU available:
    python src/training/train_stage2_sftonly.py
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset

BASE_MODEL = "LiquidAI/LFM2.5-230M"
TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/val.jsonl"
OUTPUT_DIR = "models/pg-stage2-sftonly"
ADAPTER_NAME = "pg-stage2-sftonly-adapter"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
LOGGING_STEPS = 10
SAVE_STEPS = 999999
MAX_LENGTH = 512
EVAL_STEPS = 25


def main():
    print("=" * 60)
    print("STAGE 2 ONLY: SFT directly on BASE (no CPT)")
    print(f"Base: {BASE_MODEL}")
    print(f"Data: {TRAIN_FILE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    print("Configuring LoRA for Stage 2 ONLY...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"\nLoading SFT data from {TRAIN_FILE}...")
    train_data = load_dataset("json", data_files=TRAIN_FILE, split="train")
    val_data = load_dataset("json", data_files=VAL_FILE, split="train") if os.path.exists(VAL_FILE) else None

    def format_chat(example):
        messages = [
            {"role": "user", "content": example["instruction"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"text": text + example["output"]}

    train_data = train_data.map(format_chat, remove_columns=train_data.column_names)
    if val_data:
        val_data = val_data.map(format_chat, remove_columns=val_data.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    output_dir = os.path.join(OUTPUT_DIR, ADAPTER_NAME)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=1,
        evaluation_strategy="steps" if val_data else "no",
        eval_steps=EVAL_STEPS if val_data else None,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
    )

    print(f"\nVal checks: every 25 steps, total ~525 steps, ~22 evaluations")

    print("\nStarting training...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nStage 2 ONLY complete! Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
