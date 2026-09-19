#!/usr/bin/env python3
"""
Stage 1: Continued Pre-Training (CPT) on Paul Graham text.

Trains LiquidAI/LFM2.5-230M on unified PG text using LoRA.
Plain text causal LM — NO chat template.
1 epoch on Colab T4 GPU.

Output: pg-stage1 LoRA adapter

Usage (Colab):
    colab run --gpu T4 python src/training/train_stage1.py

Or locally if GPU available:
    python src/training/train_stage1.py
"""
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset

MODEL_NAME = "LiquidAI/LFM2.5-230M"
TRAIN_FILE = "data/processed/train_cpt.txt"
OUTPUT_DIR = "models/pg-stage1"
ADAPTER_NAME = "pg-stage1-adapter"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
LOGGING_STEPS = 10
MAX_LENGTH = 512


def load_dataset(filepath: str) -> Dataset:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    print(f"Loaded {len(blocks)} text blocks from {filepath}")
    return Dataset.from_dict({"text": blocks})


def tokenize_fn(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def main():
    print("=" * 60)
    print("STAGE 1: Continued Pre-Training (CPT)")
    print(f"Model: {MODEL_NAME}")
    print(f"Data: {TRAIN_FILE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    print("Applying LoRA configuration...")
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

    print(f"\nLoading training data from {TRAIN_FILE}...")
    raw_dataset = load_dataset(TRAIN_FILE)

    tokenized = raw_dataset.map(
        lambda x: tokenize_fn(x, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=["text"],
    )

    print("Loading val CPT data for evaluation...")
    val_raw = load_dataset(str(Path(TRAIN_FILE).parent / "val_cpt.txt"))
    val_tokenized = val_raw.map(
        lambda x: tokenize_fn(x, tokenizer, MAX_LENGTH),
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

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
        save_steps=999999,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=100,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nStage 1 complete! Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
