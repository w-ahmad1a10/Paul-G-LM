#!/usr/bin/env python3
"""
PaulG-LM Colab Pipeline Runner
==============================
Runs the entire PaulG-LM pipeline on a Colab T4 GPU VM.

PREREQUISITES (on Colab VM):
1. Data files present: data/processed/, data/eval/test_questions.jsonl, data/raw/fewshot_examples.jsonl
2. Python deps: transformers, peft, datasets, accelerate

USAGE (on Colab VM):
    python src/training/run_on_colab.py

USAGE (transparent from local WSL):
    colab exec -s <session> -f /path/to/src/training/run_on_colab.py
    (may need periodic checks due to 10s poll timeout — use heartbeat output)
"""
import os
import sys
import time
import threading
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STOP_EVENT = threading.Event()


def heartbeat():
    while not STOP_EVENT.wait(5.0):
        print("⏳ running...", flush=True)


def check_data():
    required = [
        "data/processed/train_cpt.txt",
        "data/processed/val_cpt.txt",
        "data/processed/train.jsonl",
        "data/processed/val.jsonl",
        "data/eval/test_questions.jsonl",
        "data/raw/fewshot_examples.jsonl",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(BASE_DIR, f))]
    if missing:
        print("MISSING DATA FILES:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)
    print("✓ All data files present")


def run_step(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    process = subprocess.Popen(cmd, cwd=BASE_DIR)
    rc = process.wait()
    if rc != 0:
        print(f"FAILED: {description} (exit code {rc})")
        STOP_EVENT.set()
        sys.exit(rc)
    print(f"DONE: {description}")


def main():
    print("=" * 60)
    print("  PAULG-LM PIPELINE RUNNER")
    print("=" * 60)
    print(f"  Working dir: {BASE_DIR}")
    print()

    check_data()

    try:
        import transformers, peft, datasets
        print("✓ Dependencies already installed")
    except ImportError:
        print("Installing dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "transformers", "peft", "datasets", "accelerate"],
            check=True,
        )
        print("✓ Dependencies installed")

    hb_thread = threading.Thread(target=heartbeat, daemon=True)

    run_step([sys.executable, "src/training/train_stage1.py"],
             "Stage 1: CPT — Pre-train on PG text (1 epoch, ~2205 steps)")

    run_step([sys.executable, "src/training/train_stage2_sftonly.py"],
             "Stage 2 ONLY: SFT directly on base (3 epochs, ~525 steps)")

    run_step([sys.executable, "src/training/train_stage2_paulglm.py"],
             "PaulG-LM Stage 2: SFT on pg-stage1 (3 epochs, ~525 steps)")

    run_step([sys.executable, "src/training/merge.py"],
             "Merge: Combine pg-stage2-paulglm + base → PaulG-LM")

    run_step([sys.executable, "src/evaluation/eval.py"],
             "Evaluation: Generate 50 responses × 6 models")

    run_step([sys.executable, "src/data/build_judge_board.py"],
             "Judge Board: Build blind comparison HTML")

    STOP_EVENT.set()

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print("=" * 60)
    print("  Outputs:")
    print("    models/pg-stage1/           — STAGE 1 ONLY adapter")
    print("    models/pg-stage2-sftonly/   — STAGE 2 ONLY adapter")
    print("    models/pg-stage2-paulglm/   — PaulG-LM Stage 2 adapter")
    print("    models/paul-g-lm/           — Final merged PaulG-LM")
    print("    experiments/responses/      — All model responses")
    print("    experiments/judge_board.html— Blind judge board")
    print("=" * 60)


if __name__ == "__main__":
    main()
