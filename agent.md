# AGENT MANUAL — PaulG-LM
Purpose: operating manual for any AI coding agent working in this repo.
Machine: Waleed's laptop | D: is source of truth
Status: v0.1 — 2026-09-18. Project initialized. Data collected, training pending.

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
│   ├── raw/                  # RAW PG TEXT (Stage 1 input + SFT data)
│   │   ├── all_pg_essays_merged.txt  # ~1.5M words, 3 sources merged
│   │   ├── all_pg_essays_merged.jsonl
│   │   ├── essays.jsonl        # 907 essay chunks (230 essays, ~4k chars each)
│   │   ├── tweets.jsonl        # 38,080 PG tweets (2010-2026)
│   │   ├── df_train.csv        # 3,144 Q&A pairs — SFT data (READY TO USE)
│   │   ├── README.md
│   │   └── SOURCES.md
│   ├── processed/            # Processed SFT datasets (if needed)
│   │   ├── train.jsonl
│   │   └── val.jsonl
├── src/
│   ├── data/
│   │   ├── build_dataset.py  # Merge & split into train/val
│   │   ├── collect.py        # Stub: data collection (done manually)
│   │   └── generate.py       # Stub: synthetic generation (NOT NEEDED — SFT data exists)
│   ├── training/
│   │   ├── train_stage1.py   # CPT on PG text (causal LM)
│   │   ├── train_stage2.py   # SFT on Q&A pairs
│   │   └── merge.py          # Merge adapter with base model
│   └── evaluation/
│       └── eval.py           # Head-to-head eval script
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
  1. RAW      : PG essays + tweets in data/raw/
                  Stage 1 uses all_pg_essays_merged.txt ONLY — plain text for CPT

  2. SYNTH    : NOT NEEDED — SFT data already exists (data/raw/df_train.csv)
                  3,144 Q&A pairs ready for Stage 2 SFT

  3. MERGE    : build_dataset.py dedupes by instruction + splits
                  -> train.jsonl + val.jsonl

  4. TRAIN STAGE 1 : Pre-train on PG text (causal LM)
                  Base: LiquidAI/LFM2.5-230M
                  Data: data/raw/all_pg_essays_merged.txt
                  Chat Template: NONE — plain text only
                  Output: pg-stage1 (HF Hub adapter repo, 3 epochs)
                  Executed on **Colab CLI** (T4 GPU)

  5. TRAIN STAGE 2 : Fine-tune on Q&A pairs (SFT)
                  Base: Stage 1 output
                  Data: train.jsonl (from df_train.csv)
                  Method: SFT with apply_chat_template
                  Output: pg-stage2 (HF Hub adapter repo, 3 epochs)
                  Executed on **Colab CLI** (T4 GPU)

  6. MERGE FINAL : Merge adapter with base model
                  Base: LiquidAI/LFM2.5-230M
                  Adapter: pg-stage2/epoch-3
                  Output: final PG-LM (full merged model)
                  Script: src/training/merge.py

  7. EVALUATE : Head-to-head comparison (see EVALUATION section below)

=======================================================================
FOLDER CONTRACT (fixed)
----------------------
  data/raw/          all_pg_essays_merged.txt + original files + df_train.csv
  data/processed/    train.jsonl, val.jsonl
                      TRAIN/VAL ARE BUILT FROM DATA — never hand-edit.
  src/data/          build_dataset.py, collect.py, generate.py (stub — not needed)
  src/training/      train_stage1.py, train_stage2.py, merge.py
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

  df_train.csv  3,144 Q&A pairs about PG essays
                Source: RNDRantoM/paul-graham-essays-qa
                Already formatted as {question, answer} — ready for SFT

  tweets.jsonl  38,080 PG tweets (2010-2026)
                Source: aaahmet/paulg-tweets

  essays.jsonl  907 essay chunks (230 essays)
                Source: aaahmet/paulg-tweets

  SOURCES: paulgraham.com essays, pg tweets (2010-2026)

  STAGE 1 USES all_pg_essays_merged.txt ONLY — plain text for causal LM pre-training.

=======================================================================
TWO-STAGE TRAINING
==================

STAGE 1: PRE-TRAINING ON RAW TEXT
---------------------------------
Base model: LiquidAI/LFM2.5-230M
Data: data/raw/all_pg_essays_merged.txt (~1.5M words)
Method: Causal LM — predict next token
Chat Template: NONE — plain text only
Output: pg-stage1 (HF Hub adapter repo, 3 epochs)
Duration: 3 epochs on **Colab CLI T4 GPU**

STAGE 2: SFT ON Q&A PAIRS
--------------------------
Base: Stage 1 output (pg-stage1 adapter)
Data: train.jsonl (from df_train.csv — already exists, no generation needed)
Method: SFT with apply_chat_template
Output: pg-stage2 (HF Hub adapter repo, 3 epochs)
Duration: 3 epochs on **Colab CLI T4 GPU**

FINAL MERGE
-----------
Base: LiquidAI/LFM2.5-230M
Adapter: pg-stage2/epoch-3
Output: final PG-LM (full merged model)
Script: src/training/merge.py

Pipeline:
  lfm-230m → [Stage 1 CPT] → pg-base → [Stage 2 SFT] → pg-lora → [merge] → PaulG-LM

=======================================================================
SFT DATA FORMAT
---------------
{"instruction": "<question>", "output": "<PG-style answer>"}
NO system field. PG identity comes from training, not prompting.

Primary SFT data: data/raw/df_train.csv (3,144 Q&A pairs from RNDRantoM/paul-graham-essays-qa)
Convert directly to JSONL for SFT — **NO data generation needed**.

=======================================================================
EVALUATION — HEAD-TO-HEAD (NO SCORES)
========================================
This project does NOT use score-based evaluation (no 0-100 ratings).
All evaluation is **head-to-head comparison**.

EVALUATION FLOW:
  1. Generate responses from each model for test questions
  2. Sub-agent judges compare per-question responses head-to-head
  3. Report total head-to-head wins

MODELS COMPARED:
  - STAGE 1 — CPT only (pre-trained on PG text, no SFT)
  - PAULGLLM — final merged model (CPT + SFT)
  - STAGE 2 ONLY — SFT directly on base (no CPT)
  - BASE — LiquidAI/LFM2.5-230M, no fine-tuning, no prompt
  - BASE + SYSTEM PROMPT — base model with system prompt: "Speak like Paul Graham"
  - BASE + SYS PROMPT + 10 EXAMPLES — base model with system prompt + 10 few-shot
    examples from Q&A data (df_train.csv)

SYSTEM PROMPT RULE:
  System prompts are for BASE models ONLY (to set baseline behavior).
  All fine-tuned models (Stage 1, Stage 2, PaulG-LM) are NEVER told to
  speak like Paul Graham in any form of prompt. They speak like him
  naturally — their natural style becomes Paul Graham's through training.

3-JUDGE SYSTEM:
  - 3 independent sub-agent judges compare each question
  - Each judge reads BOTH responses and declares a winner (Model A or Model B)
  - The 3 judges differ ONLY in the context/prompting they receive
    (different angles of evaluation — e.g. helpfulness, naturalness,
    domain understanding)
  - CRITICAL: Judges are NEVER told which response belongs to which model
    Responses are labeled anonymously: "Response A" and "Response B"
    This prevents leakage / bias toward either model
  - Per-question winner = majority vote of 3 judges
  - If no majority (1-1-1), question is recorded as a tie

JUDGE CONTEXTS (3 different contexts — each judge gets a different framing):
  - Judge 1 context: "Which response is more helpful and accurate to the user?"
  - Judge 2 context: "Which response sounds more natural and human-written?"
  - Judge 3 context: "Which response demonstrates deeper domain understanding?"
  (Contexts may be adjusted per eval run)

OUTPUT FORMAT — head_to_head_results.jsonl:
  {"question": "...",
   "model1_name": "...",
   "model2_name": "...",
   "model1_response": "...",
   "model2_response": "...",
   "judge1_winner": "model1"|"model2"|"tie",
   "judge2_winner": "model1"|"model2"|"tie",
   "judge3_winner": "model1"|"model2"|"tie",
   "final_winner": "model1"|"model2"|"tie"}

REPORT FORMAT:
  - Per-question winner (Model A, Model B, or Tie)
  - Total head-to-head: Model A wins / Model B wins / Ties
  - Per-judge breakdown (did all 3 agree, or split?)
  - Compared models labeled clearly (STAGE 1, PAULGLLM, STAGE 2 ONLY, BASE, etc.)

=======================================================================
FOLDER CONTRACT — EVALUATION
-----------------------------
  data/eval/         Evaluation data
                     ├── test_questions.jsonl    — fixed test questions
                     ├── model1_responses.jsonl   — PG-LM (or other model) responses
                     ├── model2_responses.jsonl   — comparison model responses
                     ├── judge_results.jsonl      — per-judge winner per Q
                     ├── head_to_head_results.jsonl — final results with all judges
                     └── judge_prompts.txt        — the 3 judge contexts used

  experiments/       Eval results, judge data, blog, etc.

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
10. Stage 1: NO chat template. Stage 2: apply_chat_template.
11. PG identity from TRAINING, not prompting.
12. ALL training happens on Colab CLI (GPU T4). Use `colab run --gpu T4`.
13. Repos stay on dev branch; never commit without asking owner.
14. No score-based evaluation (0-100). Always head-to-head comparisons only.
15. Judges (sub-agents) are NEVER told which response belongs to which model.
16. All 3 judges must differ in context/prompting to ensure independent judgment.
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
  python src/training/merge.py
  colab run --gpu T4 python src/training/train_stage1.py
  colab run --gpu T4 python src/training/train_stage2.py
  colab run --gpu T4 python src/evaluation/eval.py
  streamlit run "D:\Workspace\01 Projects\w1a-board\src\app.py"

=======================================================================
END OF AGENT MANUAL — PaulG-LM v0.1
