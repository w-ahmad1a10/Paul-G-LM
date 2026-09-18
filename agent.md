# AGENT MANUAL — PaulG-LM
Purpose: operating manual for any AI coding agent working in this repo.
Machine: Waleed's laptop | D: is source of truth
Status: v0.1 — 2026-09-18. Project initialized. Data prepared, training pending.

=======================================================================
WHAT THIS PROJECT IS
-------------------
**PaulG-LM** — a small language model fine-tuned to answer ANY question
in Paul Graham's writing style (style-transfer, not fact modelling).

The model's identity is baked in through TWO-STAGE training:
  - Stage 1: Pre-train on real PG text → learns PG's voice
  - Stage 2: SFT on Q&A pairs → learns to answer questions

NO system prompt tells the model "you are Paul Graham" — that identity
comes from the training data, making it core to the model, not a prompt.

**System prompts are for BASE models only.** Fine-tuned models will speak
like Paul Graham naturally — their natural style becomes Paul Graham's.
Fine-tuned models are NEVER told to speak like PG in any form of prompt.

Project name: PaulG-LM
Repo: /mnt/c/Workspace/01 Projects/Paul-G-LM/
Base model: LiquidAI/LFM2.5-230M

=======================================================================
COMPLETE FOLDER STRUCTURE
-------------------------
Paul-G-LM/
├── agent.md                  # THIS FILE — operating manual
├── Paul-G-LM.md              # Project overview
├── .gitignore
├── data/
│   ├── raw/                  # RAW PG TEXT + EVAL DATA
│   │   ├── all_pg_essays_merged.txt  # ~1.5M words, Stage 1 CPT text
│   │   ├── all_pg_essays_merged.jsonl
│   │   ├── essays.jsonl        # 907 essay chunks (230 essays)
│   │   ├── tweets.jsonl        # 38,080 PG tweets (2010-2026)
│   │   ├── fewshot_examples.jsonl    # 10 longest val.jsonl answers
│   │   ├── README.md
│   │   └── SOURCES.md
│   ├── processed/            # PROCESSED TRAINING DATA
│   │   ├── train_cpt.txt     # Unified PG text for CPT (~14.5 MB, 35,271 blocks)
│   │   ├── val_cpt.txt       # 10% val CPT text (~1.6 MB, 3,919 blocks)
│   │   ├── train.jsonl       # SFT training (2,785 samples)
│   │   └── val.jsonl         # SFT validation (309 samples)
│   └── eval/
│       └── test_questions.jsonl  # 50 eval questions
├── src/
│   ├── data/
│   │   ├── build_dataset.py  # CSV → JSONL + 90/10 split (reproducibility)
│   │   ├── prepare_fewshot.py # Extract 10 longest val.jsonl → fewshot_examples.jsonl
│   │   ├── build_judge_board.py # Build HTML judge board for manual evaluation
│   │   ├── collect.py        # Stub: data collection (done manually)
│   │   └── generate.py       # Stub: synthetic generation (NOT NEEDED)
│   ├── training/
│   │   ├── train_stage1.py   # CPT on PG text (1 epoch, Colab T4)
│   │   ├── train_stage2.py   # SFT on Q&A pairs (3 epochs, Colab T4)
│   │   └── merge.py          # Merge adapter with base model
│   └── evaluation/
│       └── eval.py           # Head-to-head eval: generate responses for all models
├── configs/
│   └── training_config.yaml
├── models/                   # LOCAL ADAPTERS
│   ├── pg-base/
│   ├── pg-stage1/
│   └── pg-stage2/
├── logs/                     # w1a-board logs
└── experiments/              # Eval results, judge data, blog, etc.

=======================================================================
PIPELINE OVERVIEW
-----------------
  1. RAW      : PG essays + tweets + Q&A pairs in data/raw/
                  Stage 1 CPT uses unified text from all 3 sources

  2. PREP     : build_dataset.py → train.jsonl + val.jsonl (already done)
                 prepare_fewshot.py → fewshot_examples.jsonl (10 longest)

  3. TRAIN STAGE 1 : Pre-train on PG text (causal LM)
                  Base: LiquidAI/LFM2.5-230M
                  Data: data/processed/train_cpt.txt
                  Chat Template: NONE — plain text only
                  Epochs: **1**
                  Output: pg-stage1 (HF Hub adapter repo)
                  Executed on **Colab CLI** (T4 GPU)

  4. TRAIN STAGE 2 : Fine-tune on Q&A pairs (SFT)
                  Base: Stage 1 output (pg-stage1 adapter)
                  Data: data/processed/train.jsonl
                  Method: SFT with apply_chat_template
                  Epochs: **3**
                  Output: pg-stage2 (HF Hub adapter repo)
                  Executed on **Colab CLI** (T4 GPU)

  5. MERGE FINAL : Merge adapter with base model
                  Base: LiquidAI/LFM2.5-230M
                  Adapter: pg-stage2/epoch-3
                  Output: final PG-LM (full merged model)
                  Script: src/training/merge.py

  Models saved:
  - Stage 1 end: pg-stage1 adapter (1 model)
  - Stage 2 end: pg-stage2 adapter (1 model)
  - Merged: PaulG-LM full model (1 model)
  - Total: **3 models** (2 trained, 1 merged)

  6. EVALUATE  : Generate responses for all models → Build Judge Board
                  See EVALUATION section below

=======================================================================
FOLDER CONTRACT (fixed)
----------------------
  data/raw/          all text sources + fewshot + test questions
  data/processed/    train_cpt.txt, val_cpt.txt, train.jsonl, val.jsonl
                      TRAIN/VAL ARE BUILT FROM DATA — never hand-edit.
  src/data/          build_dataset.py, prepare_fewshot.py, build_judge_board.py,
                     collect.py, generate.py (stub)
  src/training/      train_stage1.py (1 epoch), train_stage2.py (3 epochs), merge.py
  src/evaluation/    eval.py
  configs/           training_config.yaml
  models/            Local adapter checkpoints
  logs/              w1a-board logs (RunLog format)
  agent.md           this file

=======================================================================
RAW DATA (data/raw/)
--------------------
  all_pg_essays_merged.txt  ~1.5M words (3 sources combined, exact dedup)
                             Real PG: essays, tweets, book excerpts

  fewshot_examples.jsonl  10 Q&A pairs with longest answers from val.jsonl
                          Used for BASE + SYS PROMPT + 10 EXAMPLES model only

  tweets.jsonl  38,080 PG tweets (2010-2026)
                Source: aaahmet/paulg-tweets

  essays.jsonl  907 essay chunks (230 essays)
                Source: aaahmet/paulg-tweets

  SOURCES: paulgraham.com essays, pg tweets (2010-2026)

  Stage 1 uses unified text combining all 3 essay/tweet sources.

=======================================================================
TWO-STAGE TRAINING
==================

STAGE 1: PRE-TRAINING ON RAW TEXT
---------------------------------
Base model: LiquidAI/LFM2.5-230M
Data: data/processed/train_cpt.txt (~1.5M words, unified from essays+tweets+merged)
Val data: data/processed/val_cpt.txt (eval every 100 steps)
Method: Causal LM — predict next token
Chat Template: NONE — plain text only
Output: pg-stage1 (HF Hub adapter repo, **1 epoch**)
Duration: **1 epoch** on **Colab CLI T4 GPU**
Total steps: **2,205** (35,271 blocks / 16 effective batch)
Val loss checks: **~23** (every 100 steps + end)

STAGE 2: SFT ON Q&A PAIRS
--------------------------
Base: Stage 1 output (pg-stage1 adapter)
Data: data/processed/train.jsonl (2,785 samples)
Val data: data/processed/val.jsonl (eval every 25 steps)
Method: SFT with apply_chat_template
Output: pg-stage2 (HF Hub adapter repo, **3 epochs**)
Duration: **3 epochs** on **Colab CLI T4 GPU**
Total steps: **525** (2,785 samples / 16 effective batch × 3 epochs)
Val loss checks: **~22** (every 25 steps + end)

FINAL MERGE
-----------
Base: LiquidAI/LFM2.5-230M
Adapter: pg-stage2/epoch-3
Output: final PG-LM (full merged model)
Script: src/training/merge.py

Pipeline:
  lfm-230m → [Stage 1 CPT, 1 epoch] → pg-base → [Stage 2 SFT, 3 epochs] → pg-lora → [merge] → PaulG-LM

=======================================================================
SFT DATA FORMAT
---------------
{"instruction": "<question>", "output": "<PG-style answer>"}
NO system field. PG identity comes from training, not prompting.

SFT training data: data/processed/train.jsonl (2,785 samples, from original CSV)
SFT validation data: data/processed/val.jsonl (309 samples)
**NO data generation needed** — already prepared.

Few-shot examples for BASE+SYS+10: data/raw/fewshot_examples.jsonl (10 samples,
extracted as 10 longest answers from val.jsonl)

=======================================================================
EVALUATION — JUDGE BOARD (NO SCORES)
==========================================
This project does NOT use score-based evaluation (no 0-100 ratings).
All evaluation is **head-to-head comparison** using a **Judge Board**.

GENERATION PARAMS (eval/inference):
  temperature: 0.9 | top_k: 50 | max_new_tokens: 256
  (No top_p, no repetition_penalty)

EVALUATION FLOW:
  1. Generate responses from ALL models for 50 test questions
     Models: STAGE 1, STAGE 2 ONLY, BASE, BASE+SYS_PROMPT, BASE+SYS+10_EXAMPLES, PAULGLLM
  2. Build Judge Board (HTML) via build_judge_board.py
  3. **You (user)** manually judge each question blind
  4. Tally results → final head-to-head report

MODELS COMPARED (PaulG-LM faces each one):
  - STAGE 1 ONLY — CPT only (1 epoch, no SFT)
  - STAGE 2 ONLY — SFT only (3 epochs, no CPT)
  - BASE — LiquidAI/LFM2.5-230M, no fine-tuning, no prompt
  - BASE + SYSTEM PROMPT — base + system prompt: "Speak like Paul Graham"
  - BASE + SYS PROMPT + 10 EXAMPLES — base + system prompt + 10 few-shot
    examples from val.jsonl (longest answers)
  - PAULGLLM — final merged model (CPT + SFT)

JUDGE BOARD (Blind):
  - All responses are labeled "Response A" and "Response B"
  - **Model identity is NEVER shown** — you don't know which is PaulG-LM
  - Order randomized per question (PaulG-LM could be A or B)
  - You pick winner per question: A, B, or Tie
  - Judge Board built as interactive HTML

SYSTEM PROMPT RULE:
  System prompts are for BASE models ONLY (to set baseline behavior).
  All fine-tuned models (Stage 1, Stage 2, PaulG-LM) are NEVER told to
  speak like Paul Graham in any form of prompt. They speak like him
  naturally — their natural style becomes Paul Graham's through training.

OUTPUT:
  - Per-question winner (PaulG-LM wins, Other model wins, or Tie)
  - Total head-to-head record per comparison pair
  - Example: "PaulG-LM vs STAGE 1 ONLY: 32 wins / 12 losses / 6 ties"

=======================================================================
NON-NEGOTIABLE RULES
--------------------
2. NO GitHub push. Transport via **Colab CLI** only (for training/runs).
3. No tokens/secrets in code.
4. w1a-board instead of wandb.
5. Owner cannot code — write ALL code, explain in plain language.
6. Diff training code before cloud runs.
7. Large datasets stay local.
8. NEVER add system prompt to training data (fine-tuned models).
9. System prompts are for BASE models ONLY.
10. Stage 1: **1 epoch**. Stage 2: **3 epochs**.
11. Stage 1: NO chat template. Stage 2: apply_chat_template.
12. PG identity from TRAINING, not prompting.
13. ALL training happens on Colab CLI (GPU T4). Use `colab run --gpu T4`.
14. Repos stay on dev branch; never commit without asking owner.
15. No score-based evaluation (0-100). Always head-to-head with Judge Board.
16. Judge Board is blind — model identity never shown to user-judge.
17. SFT data already exists — no data generation needed.

=======================================================================
COLAB CLI WORKFLOW
--------------------
1. Authenticate: `colab auth`
2. For training runs: `colab run --gpu T4 python src/training/train_stage1.py`
3. For interactive dev: `colab new -s pg-train --gpu T4 --high-mem`
4. Run local scripts on VM: `colab exec -s pg-train -f /home/waleed10/.../train.py`
5. Upload data: `colab upload -s pg-train ./data ./data`
6. Download results: `colab download -s pg-train /content/model ./model`
7. Stop when done: `colab stop -s pg-train`

Note: Colab CLI must be installed from git:
  uv tool install google-colab-cli

=======================================================================
KEY COMMANDS
--------------
  python src/data/build_dataset.py
  python src/data/prepare_fewshot.py
  python src/data/build_judge_board.py
  python src/training/merge.py
  colab run --gpu T4 python src/training/train_stage1.py
  colab run --gpu T4 python src/training/train_stage2.py
  colab run --gpu T4 python src/evaluation/eval.py
  streamlit run "D:\Workspace\01 Projects\w1a-board\src\app.py"

=======================================================================
END OF AGENT MANUAL — PaulG-LM v0.1
