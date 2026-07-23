#!/usr/bin/env bash
# Run BBT fine-tuning sweeps from the FGNO package root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export FGNO_WANDB="${FGNO_WANDB:-0}"

echo "FGNO root: $ROOT"
echo "========================================"
echo "Speech fine-tuning"
echo "========================================"
python finetune/BBT_finetune/run_finetuning_speech.py

echo "========================================"
echo "Pitch fine-tuning"
echo "========================================"
python finetune/BBT_finetune/run_finetuning_pitch.py

echo "========================================"
echo "Volume fine-tuning"
echo "========================================"
python finetune/BBT_finetune/run_finetuning_volume.py

echo "========================================"
echo "Sentence fine-tuning"
echo "========================================"
python finetune/BBT_finetune/run_finetuning_sentence.py

echo "All BBT fine-tuning tasks completed."
