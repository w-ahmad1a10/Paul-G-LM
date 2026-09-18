# Paul-G-LM

Fine-tune a small language model to answer any question in Paul Graham's writing style.

## Project Structure

```
Paul-G-LM/
├── agent.md              # Operating manual for AI agents
├── Paul-G-LM.md          # This file — project overview
├── .gitignore
├── data/
│   ├── raw/              # PG text data (Stage 1 input + SFT data + eval)
│   │   ├── all_pg_essays_merged.txt   # Stage 1 CPT text (merged essays)
│   │   ├── all_pg_essays_merged.jsonl
│   │   ├── essays.jsonl        # 907 essay chunks (230 essays)
│   │   ├── tweets.jsonl        # 38,080 PG tweets (2010-2026)
│   │   ├── fewshot_examples.jsonl    # 10 longest val.jsonl examples (for BASE+SYS+10)
│   │   ├── README.md
│   │   └── SOURCES.md
│   ├── processed/        # Processed datasets for training
│   │   ├── train_cpt.txt   # Unified PG text for Stage 1 CPT (~14.5 MB)
│   │   ├── val_cpt.txt     # 10% val CPT text (~1.6 MB)
│   │   ├── train.jsonl     # SFT training data (2,785 samples)
│   │   └── val.jsonl       # SFT validation data (309 samples)
│   └── eval/
│       └── test_questions.jsonl  # 50 eval questions (longest answers from original)
├── src/
│   ├── data/             # Data processing scripts
│   ├── training/         # Fine-tuning scripts
│   └── evaluation/       # Evaluation scripts
├── configs/              # Training configs
├── models/               # Local adapter checkpoints
└── logs/                 # w1a-board logs
```

## Pipeline

1. **Data**: PG essays + tweets + Q&A pairs (all in `data/raw/`, already collected)
2. **Stage 1 (CPT)**: Pre-train on unified PG text → learns PG's voice (1 epoch)
3. **Stage 2 (SFT)**: Fine-tune on Q&A pairs → learns to answer questions (3 epochs)
4. **Merge**: Merge adapter with base model → final PG-LM
5. **Evaluate**: Judge Board — you compare head-to-head (no scores)

## SFT Data

SFT data already exists: `data/raw/df_train.csv` was processed into `data/processed/train.jsonl` + `val.jsonl`. **No data generation needed.**

## Base Model

LiquidAI/LFM2.5-230M

## Training

All training runs on **Google Colab** via Colab CLI.

- Stage 1: `colab run --gpu T4 python src/training/train_stage1.py` (1 epoch)
- Stage 2: `colab run --gpu T4 python src/training/train_stage2.py` (3 epochs)

## Evaluation — Judge Board

Head-to-head comparison only. No score-based evaluation.

**Generation:** temperature=0.9, top_k=50, max_new_tokens=256

**Models compared:**
- STAGE 1 ONLY — CPT only (pre-trained on PG text, no SFT)
- STAGE 2 ONLY — SFT directly on base (no CPT)
- BASE — LiquidAI/LFM2.5-230M, no fine-tuning, no prompt
- BASE + SYS PROMPT — base + system prompt "Speak like Paul Graham"
- BASE + SYS PROMPT + 10 EXAMPLES — base + system prompt + 10 few-shot from val.jsonl (longest answers)
- PAULGLLM — final merged model (CPT + SFT)

**Judge Board:** You manually judge 50 questions blind. Responses labeled "Response A" / "Response B" — model identity hidden. You pick winner per question. PaulG-LM always faces each comparison model separately.

## Differences from Style-Transfer Projects

- Same two-stage training pipeline as standard style-transfer approaches
- No score-based evaluation — always head-to-head with Judge Board
- All training on Colab CLI
- SFT data already exists — no generation phase
- System prompts only for base models; fine-tuned models speak naturally

## Quick Start

```bash
# Data prep
python src/data/build_dataset.py
python src/data/prepare_fewshot.py

# Stage 1 (Colab, 1 epoch)
colab run --gpu T4 python src/training/train_stage1.py

# Stage 2 (Colab, 3 epochs)
colab run --gpu T4 python src/training/train_stage2.py

# Merge final model
python src/training/merge.py

# Build Judge Board
python src/data/build_judge_board.py

# Generate responses for eval
colab run --gpu T4 python src/evaluation/eval.py
```
