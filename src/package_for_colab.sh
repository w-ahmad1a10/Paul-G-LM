#!/bin/bash
# Package all files needed for Colab run.
# Run this locally, then follow the printed instructions.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT="/tmp/paulglim_colab.tar.gz"

cd "$PROJECT_DIR"

tar czf "$OUTPUT" \
  --exclude='models/*' \
  --exclude='logs/*' \
  --exclude='.git' \
  --exclude='data/raw/all_pg_essays_merged.*' \
  --exclude='data/raw/essays.jsonl' \
  --exclude='data/raw/tweets.jsonl' \
  --exclude='data/raw/train_filtered.csv' \
  --exclude='data/raw/README.md' \
  --exclude='data/raw/SOURCES.md' \
  src configs data/processed data/eval/test_questions.jsonl \
  data/raw/fewshot_examples.jsonl \
  Paul-G-LM.md agent.md

echo ""
echo "Created: $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  RUN ON COLAB — Step by step"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "1. Provision VM:"
echo "   colab new -s paulglim --gpu T4"
echo ""
echo "2. Upload package:"
echo "   colab upload -s paulglim $OUTPUT /content/paulglim_colab.tar.gz"
echo ""
echo "3. Install deps (uv via Colab CLI):"
echo "   colab install -s paulglim transformers peft datasets accelerate"
echo ""
echo "4. Extract data tarball on VM:"
echo "   colab exec -s paulglim -f /tmp/setup_colab.py"
echo ""
echo "5. Run full pipeline:"
echo "   colab exec -s paulglim -f /mnt/c/Workspace/01 Projects/Paul-G-Lm/src/training/run_on_colab.py"
echo "   (may time out due to 10s poll — check progress with:)"
echo "   colab exec -s paulglim 'ps aux | grep python'"
echo ""
echo "6. Download results:"
echo "   colab download -s paulglim /content/models ./paulglim_models"
echo "   colab download -s paulglim /content/experiments ./paulglim_experiments"
echo ""
echo "7. Build judge board locally (uses downloaded responses):"
echo "   cd \"$PROJECT_DIR\" && python src/data/build_judge_board.py"
echo ""
echo "8. Stop VM when done:"
echo "   colab stop -s paulglim"
echo ""
echo "═══════════════════════════════════════════════════════════"
