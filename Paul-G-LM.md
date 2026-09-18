# Paul-G-LM

Fine-tune a small language model to answer any question in Paul Graham's writing style.

## Project Structure

```
Paul-G-LM/
├── agent.md              # Operating manual for AI agents
├── Paul-G-LM.md          # This file — project overview
├── .gitignore
├── data/
│   ├── raw/              # PG text data (Stage 1 input)
│   │   ├── all_pg_essays_merged.txt
│   │   ├── all_pg_essays_merged.jsonl
│   │   ├── essays.jsonl
│   │   ├── tweets.jsonl
│   │   ├── df_train.csv
│   │   ├── README.md
│   │   └── SOURCES.md
│   └── processed/        # Processed SFT datasets
├── src/
│   ├── data/             # Data processing scripts
│   ├── training/         # Fine-tuning scripts
│   └── evaluation/       # Evaluation scripts
├── configs/              # Training configs
├── models/               # Local adapter checkpoints
└── logs/                 # w1a-board logs
```

## Pipeline

1. **Data Collection**: PG essays, tweets, Q&A pairs (already collected in `data/raw/`)
2. **Stage 1 (CPT)**: Pre-train on PG essays/tweets → learns PG's voice
3. **Stage 2 (SFT)**: Fine-tune on Q&A pairs → learns to answer questions
4. **Merge**: Merge adapter with base model → final PG-LM
5. **Evaluate**: Test if model sounds like Paul Graham

## Base Model

LiquidAI/LFM2.5-230M (same as TrumpLM)

## Training

All training runs on **Google Colab** via Colab CLI.

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

# Evaluate
python src/evaluation/eval.py
```
