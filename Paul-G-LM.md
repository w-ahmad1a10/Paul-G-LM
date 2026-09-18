# Paul-G-LM

Fine-tune a small language model to answer any question in Paul Graham's writing style.

## Project Structure

```
Paul-G-LM/
├── agent.md              # Operating manual for AI agents
├── Paul-G-LM.md          # This file — project overview
├── .gitignore
├── data/
│   ├── raw/              # PG text data (Stage 1 input + SFT data)
│   │   ├── all_pg_essays_merged.txt
│   │   ├── all_pg_essays_merged.jsonl
│   │   ├── essays.jsonl
│   │   ├── tweets.jsonl
│   │   ├── df_train.csv        # 3,144 Q&A pairs — SFT data (already exists)
│   │   ├── README.md
│   │   └── SOURCES.md
│   └── processed/        # Processed SFT datasets (if needed)
├── src/
│   ├── data/             # Data processing scripts
│   ├── training/         # Fine-tuning scripts
│   └── evaluation/       # Evaluation scripts
├── configs/              # Training configs
├── models/               # Local adapter checkpoints
└── logs/                 # w1a-board logs
```

## Pipeline

1. **Data**: PG essays + tweets for Stage 1 CPT, Q&A pairs for Stage 2 SFT (already collected in `data/raw/`)
2. **Stage 1 (CPT)**: Pre-train on PG essays/tweets → learns PG's voice
3. **Stage 2 (SFT)**: Fine-tune on Q&A pairs → learns to answer questions
4. **Merge**: Merge adapter with base model → final PG-LM
5. **Evaluate**: Head-to-head comparison (no scores)

## SFT Data

SFT data already exists: `data/raw/df_train.csv` (3,144 Q&A pairs about PG essays, from `RNDRantoM/paul-graham-essays-qa`). **No data generation needed.**

## Base Model

LiquidAI/LFM2.5-230M

## Training

All training runs on **Google Colab** via Colab CLI.

## Evaluation

Head-to-head comparison only. No score-based evaluation.

**Models compared:**
- STAGE 1 — CPT only (pre-trained on PG text, no SFT)
- PAULGLLM — final merged model (CPT + SFT)
- STAGE 2 ONLY — SFT directly on base (no CPT)
- BASE — LiquidAI/LFM2.5-230M, no fine-tuning
- BASE + SYSTEM PROMPT — base model with system prompt "Speak like Paul Graham"
- BASE + SYS PROMPT + 10 EXAMPLES — base model with system prompt + 10 few-shot examples from Q&A data

Per question, sub-agent judges compare responses head-to-half. 3 judges, different contexts, anonymous labels.

## Differences from Style-Transfer Projects

- Same two-stage training pipeline as standard style-transfer approaches
- No score-based evaluation — always head-to-head
- 3-judge system with leakage prevention (judges see responses anonymously)
- All training on Colab CLI
- SFT data already exists — no generation phase

## Quick Start

```bash
# Data prep
python src/data/build_dataset.py

# Stage 1 (Colab)
colab run --gpu T4 python src/training/train_stage1.py

# Stage 2 (Colab)
colab run --gpu T4 python src/training/train_stage2.py

# Merge final model
python src/training/merge.py

# Evaluate (head-to-head)
colab run --gpu T4 python src/evaluation/eval.py
```
