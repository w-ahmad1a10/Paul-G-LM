"""
Build dataset for Stage 2 SFT from CSV source.

Reads data/raw/train_filtered.csv, converts to JSONL format
{instruction, output}, shuffles with fixed seed, splits 90/10
into data/processed/train.jsonl and data/processed/val.jsonl.

This is a reproducibility script — data was already prepared manually,
but this script ensures anyone can recreate the splits from the CSV.

Usage: python src/data/build_dataset.py
         python src/data/build_dataset.py --val 0.2 --seed 7
"""
import argparse
import csv
import json
import random
from pathlib import Path

RAW_CSV = Path("data/raw/train_filtered.csv")
TRAIN_JSONL = Path("data/processed/train.jsonl")
VAL_JSONL = Path("data/processed/val.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("Loading CSV...")
    with open(RAW_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Total rows: {len(rows)}")

    samples = []
    for row in rows:
        samples.append({
            "instruction": row["question"].strip(),
            "output": row["answer"].strip(),
        })

    random.Random(args.seed).shuffle(samples)
    n_val = int(len(samples) * args.val)
    val_rows, train_rows = samples[:n_val], samples[n_val:]

    TRAIN_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(VAL_JSONL, "w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"train.jsonl: {len(train_rows)} samples")
    print(f"val.jsonl: {len(val_rows)} samples")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
