"""
Extract 10 longest answers from val.jsonl as few-shot examples.

Reads data/processed/val.jsonl, sorts by output length descending,
takes top 10, writes to data/raw/fewshot_examples.jsonl.

These 10 examples are used for BASE + SYS PROMPT + 10 EXAMPLES model
in the head-to-head evaluation.

Usage: python src/data/prepare_fewshot.py
"""
import json
import os
from pathlib import Path

VAL_JSONL = Path("data/processed/val.jsonl")
OUTPUT = Path("data/raw/fewshot_examples.jsonl")
N_EXAMPLES = 10


def main():
    print("Loading val.jsonl...")
    with open(VAL_JSONL, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    print(f"Total samples: {len(samples)}")

    sorted_samples = sorted(samples, key=lambda s: len(s.get("output", "")), reverse=True)
    top_n = sorted_samples[:N_EXAMPLES]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for sample in top_n:
            obj = {
                "instruction": sample["instruction"],
                "output": sample["output"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(top_n)} few-shot examples to {OUTPUT}")
    for i, s in enumerate(top_n):
        print(f"  {i+1}. len={len(s['output'])} chars | {s['instruction'][:60]}...")


if __name__ == "__main__":
    main()
