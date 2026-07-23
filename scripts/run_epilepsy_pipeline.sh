#!/usr/bin/env bash
# End-to-end Epilepsy pipeline: prep data -> pretrain -> finetune -> results.
#
# Usage:
#   bash scripts/run_epilepsy_pipeline.sh
#
# Skips a stage if its output already exists. Set FGNO_WANDB=0 to disable
# W&B logging (on by default). Requires data_preprocess/epilepsy/epilepsy_srcdata.csv
# (included) -- see data_preprocess/README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

if [[ ! -f data/epilepsy_raw/train.pt ]]; then
  echo "[1/4] Building data/epilepsy_raw from epilepsy_srcdata.csv ..."
  python data_preprocess/epilepsy/data_preprocess_epilepsy.py
else
  echo "[1/4] data/epilepsy_raw already present, skipping."
fi

if [[ ! -f data/epilepsy/train_stft.pt ]]; then
  echo "[2/4] STFT-transforming data/epilepsy_raw -> data/epilepsy ..."
  python data_preprocess/epilepsy/preprocess_stft_epilepsy.py
else
  echo "[2/4] data/epilepsy already present, skipping."
fi

if [[ ! -f checkpoints/ffm_epilepsy_pretrain.pt ]]; then
  echo "[3/4] Pretraining FFM backbone on epilepsy STFT data ..."
  python pretrain/pretrain_epilepsy.py
else
  echo "[3/4] checkpoints/ffm_epilepsy_pretrain.pt already present, skipping."
fi

echo "[4/4] Fine-tuning + evaluating (layer x extraction-time sweep) ..."
python finetune/finetune_epilepsy.py

echo "Done. Results: outputs/epilepsy_clean_input_acc_f1.csv"
