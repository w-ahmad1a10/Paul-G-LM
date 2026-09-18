#!/usr/bin/env python3
"""
Head-to-head evaluation: generate responses for all models.

Generates 50 responses from each model (STAGE 1, STAGE 2 ONLY, BASE,
BASE+SYS_PROMPT, BASE+SYS_PROMPT+10_EXAMPLES, PAULGLLM) for the test
questions. Saves responses for Judge Board comparison.

Models are compared against PaulG-LM (M6) head-to-head.

Usage (Colab):
    colab run --gpu T4 python src/evaluation/eval.py

Usage (local with GPU):
    python src/evaluation/eval.py
"""
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "LiquidAI/LFM2.5-230M"
TEST_QUESTIONS = "data/eval/test_questions.jsonl"
OUTPUT_DIR = "experiments/responses"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.9
TOP_K = 50


def load_test_questions(filepath: str) -> list:
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            questions.append(obj["instruction"])
    return questions


def generate_response(model, tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full[len(prompt):].strip()
    return response


def generate_all_responses(questions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    questions_file = os.path.join(output_dir, "questions.jsonl")
    with open(questions_file, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps({"question": q}) + "\n")
    print(f"Wrote {len(questions)} questions to {questions_file}")


def main():
    print("=" * 60)
    print("HEAD-TO-HEAD EVALUATION: Response Generation")
    print(f"Test questions: {TEST_QUESTIONS}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    questions = load_test_questions(TEST_QUESTIONS)
    print(f"\nLoaded {len(questions)} test questions")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    models_to_generate = {
        "stage1_only": None,
        "stage2_only": None,
        "base": None,
        "base_sys_prompt": None,
        "base_sys_10examples": None,
        "paulgllm": None,
    }

    for model_key in models_to_generate:
        out_file = os.path.join(OUTPUT_DIR, f"{model_key}_responses.jsonl")
        if os.path.exists(out_file):
            print(f"\n{model_key}: responses already exist at {out_file}")
            continue

        print(f"\nGenerating responses for {model_key}...")

        if model_key == "base":
            print("  Loading base model (no fine-tuning)...")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        elif model_key == "paulgllm":
            print("  Loading PaulG-LM (merged model)...")
            model_path = "models/paul-g-lm"
            if os.path.exists(model_path):
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                print(f"  ERROR: Merged model not found at {model_path}")
                print("  Run merge.py first.")
                continue

        elif model_key == "stage1_only":
            print("  Loading base + Stage 1 adapter...")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = PeftModel.from_pretrained(model, "models/pg-stage1/pg-stage1-adapter")

        elif model_key == "stage2_only":
            print("  Loading base + Stage 2 adapter (no Stage 1)...")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = PeftModel.from_pretrained(model, "models/pg-stage2/pg-stage2-adapter")

        elif model_key == "base_sys_prompt":
            print("  Loading base model with system prompt...")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        elif model_key == "base_sys_10examples":
            print("  Loading base model with system prompt + 10 examples...")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        responses = []
        for i, question in enumerate(questions):
            print(f"  [{i+1}/{len(questions)}] {question[:60]}...")

            if model_key in ("base_sys_prompt", "base_sys_10examples"):
                sys_prompt = "You are Paul Graham. Speak in your natural writing style."
                if model_key == "base_sys_10examples":
                    with open("data/raw/fewshot_examples.jsonl", "r") as f:
                        examples = [json.loads(line) for line in f if line.strip()]
                    examples_text = "\n\n".join(
                        [f"Q: {ex['instruction']}\nA: {ex['output']}" for ex in examples]
                    )
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": examples_text + "\n\nQuestion: " + question},
                    ]
                else:
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": question},
                    ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt")
            else:
                messages = [{"role": "user", "content": question}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt")

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            full = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full[len(prompt):].strip()
            responses.append({
                "question": question,
                "response": response,
                "model": model_key,
            })

        out_file = os.path.join(OUTPUT_DIR, f"{model_key}_responses.jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for r in responses:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(responses)} responses to {out_file}")

    print("\n" + "=" * 60)
    print("EVALUATION RESPONSES COMPLETE")
    print(f"All files in: {OUTPUT_DIR}/")
    files = sorted(os.listdir(OUTPUT_DIR))
    for f in files:
        if f.endswith(".jsonl"):
            path = os.path.join(OUTPUT_DIR, f)
            count = sum(1 for _ in open(path))
            print(f"  {f}: {count} responses")
    print("=" * 60)
    print("\nNext: run build_judge_board.py to build the blind Judge Board.")


if __name__ == "__main__":
    main()
