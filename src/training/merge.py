#!/usr/bin/env python3
"""
Merge Stage 2 LoRA adapter with base model → final PaulG-LM.

Downloads LiquidAI/LFM2.5-230M + pg-stage2 adapter,
merges them into a single model, and saves the full model.

For Colab: run on local or Kaggle after downloading adapter to Colab.

Usage: python src/training/merge.py
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "LiquidAI/LFM2.5-230M"
STAGE2_ADAPTER = "models/pg-stage2/pg-stage2-adapter"
OUTPUT_DIR = "models/paul-g-lm"


def main():
    print("=" * 60)
    print("MERGE: PaulG-LM Final Model")
    print(f"Base: {BASE_MODEL}")
    print(f"Adapter: {STAGE2_ADAPTER}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    if not os.path.exists(STAGE2_ADAPTER):
        print(f"\nERROR: Stage 2 adapter not found at {STAGE2_ADAPTER}")
        print("Run Stage 2 training first: colab run --gpu T4 python src/training/train_stage2.py")
        return

    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Stage 2 adapter...")
    model = PeftModel.from_pretrained(base_model, STAGE2_ADAPTER)

    print("Merging adapter with base model...")
    merged = model.merge_and_unload()
    print("Merge complete!")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nFinal model saved to: {OUTPUT_DIR}")
    print(f"Use: AutoModelForCausalLM.from_pretrained('{OUTPUT_DIR}')")


if __name__ == "__main__":
    main()
