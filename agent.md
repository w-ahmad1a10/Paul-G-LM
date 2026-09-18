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

Project name: PaulG-LM
Repo: /mnt/c/Workspace/01 Projects/Paul-G-LM/
Base model: LiquidAI/LFM2.5-230M (same as TrumpLM)

=======================================================================
COMPLETE FOLDER STRUCTURE
-------------------------
Paul-G-LM/
├── agent.md                  # THIS FILE — operating manual
├── Paul-G-LM.md              # Project overview
├── .gitignore
├── data/
│   ├── raw/                  # RAW PG TEXT (Stage 1 input)
│   │   ├── all_pg_essays_merged.txt  # ~1.5M words, 3 sources merged
│   │   ├── all_pg_essays_merged.jsonl
│   │   ├── essays.jsonl        # 907 essay chunks (230 essays, ~4k chars each)
│   │   ├── tweets.jsonl        # 38,080 PG tweets (2010-2026)
│   │   ├── df_train.csv        # 3,144 Q&A pairs (AI-generated about PG essays)
│   │   ├── README.md
│   │   └── SOURCES.md
│   ├── processed/            # SYNTHETIC SFT DATA (if generated beyond df_train)
│   │   ├── train.jsonl
│   │   └── val.jsonl
├── src/
│   ├── data/
│   │   ├── build_dataset.py  # Merge & split into train/val
│   │   ├── collect.py        # Stub: data collection (done manually)
│   │   └── generate.py       # Stub: synthetic generation (if needed)
│   ├── training/
│   │   ├── train_stage1.py   # CPT on PG text (causal LM)
│   │   ├── train_stage2.py   # SFT on Q&A pairs
│   │   └── merge.py          # Merge adapter with base model
│   └── evaluation/
│       └── eval.py           # Generate responses + save for grading
├── configs/
│   └── training_config.yaml
├── models/                   # LOCAL ADAPTERS
│   ├── pg-base/
│   ├── pg-stage1/
│   └── pg-stage2/
├── logs/                     # w1a-board logs
└── experiments/              # Eval results, blog, etc.

=======================================================================
PIPELINE OVERVIEW
-----------------
  1. RAW      : PG essays + tweets in data/raw/
                  Stage 1 uses all_pg_essays_merged.txt ONLY — plain text for CPT

  2. SYNTH    : (if needed) generate Q&A pairs in PG style
                  -> data/processed/batch_*.jsonl
                  System prompt REMOVED — identity comes from training

  3. MERGE    : build_dataset.py dedupes by instruction + splits
                  -> train.jsonl + val.jsonl

  4. TRAIN STAGE 1 : Pre-train on PG text (causal LM)
                  Base: LiquidAI/LFM2.5-230M
                  Data: data/raw/all_pg_essays_merged.txt
                  Chat Template: NONE — plain text only
                  Output: pg-stage1 adapter (HF Hub)
                  Executed on **Colab CLI** (T4 GPU)

  5. TRAIN STAGE 2 : Fine-tune on Q&A pairs (SFT)
                  Base: Stage 1 output
                  Data: train.jsonl
                  Method: SFT with apply_chat_template
                  Output: pg-stage2 adapter (HF Hub)
                  Executed on **Colab CLI** (T4 GPU)

  6. MERGE FINAL : Merge adapter with base model
                  Base: LiquidAI/LFM2.5-230M
                  Adapter: pg-stage2/epoch-3
                  Output: final PG-LM (full merged model)
                  Script: src/training/merge.py

  7. EVALUATE : eval.py generates responses from model
                  Reference model: lfm-230m
                  External agent grades 1-5

=======================================================================
FOLDER CONTRACT (fixed)
----------------------
  data/raw/          all_pg_essays_merged.txt + original files
  data/processed/    train.jsonl, val.jsonl
                      TRAIN/VAL ARE BUILT FROM DATA — never hand-edit.
  src/data/          build_dataset.py, collect.py, generate.py
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
Data: train.jsonl (from df_train.csv or generated batches)
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
Can be directly converted to JSONL for SFT — may skip synthetic generation entirely.

=======================================================================
NON-NEGOTIABLE RULES
--------------------
1. All toolchains under D:\Miniconda3 / ~/miniconda3
2. NO GitHub push. Transport via **Colab CLI** only (for training/runs).
3. No tokens/secrets in code.
4. w1a-board instead of wandb.
5. Owner cannot code — write ALL code, explain in plain language.
6. Diff training code before cloud runs.
7. Large datasets stay local.
8. NEVER add system prompt to training data.
9. Stage 1: NO chat template. Stage 2: apply_chat_template.
10. PG identity from TRAINING, not prompting.
11. ALL training happens on Colab CLI (GPU T4). Use `colab run --gpu T4`.
12. Repos stay on dev branch; never commit without asking owner.

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
  python src/evaluation/eval.py
  streamlit run "D:\Workspace\01 Projects\w1a-board\src\app.py"
  colab run --gpu T4 python src/training/train_stage1.py
  colab run --gpu T4 python src/training/train_stage2.py

=======================================================================
END OF AGENT MANUAL — PaulG-LM v0.1
