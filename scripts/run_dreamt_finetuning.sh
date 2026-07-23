#!/usr/bin/env bash
# Run all DREAMT fine-tuning experiments (FFM + MAE, sleep-stage classification
# + skin-temperature regression) from the FGNO package root.
#
# Usage:
#   bash scripts/run_dreamt_finetuning.sh
#
# Prerequisites (see data_preprocess/dreamt/README.md and the "DREAMT" section
# of the top-level README for how to produce these):
#   data/dreamt/processed_BVP_ACC_up_sample_datasets/
#     finetune_{train,val,test}_dataset_sub34.pt        (preprocess_dreamt_subject_splits.py)
#     finetune_skin_{train,val,test}_dataset_sub34.pt
#   checkpoints/ffm_dreamt_BVP_ACC_upsample.pt          (pretrain/pretrain_ffm_dreamt.py)
#   checkpoints/mae_dreamt_BVP_ACC_upsample_epoch_250.pt (pretrain/pretrain_mae_dreamt.py)
#
# Each script sweeps encoder layer x feature-extraction time, trains a linear
# probe per combination, and writes results to outputs/dreamt_*.csv. Set
# FGNO_WANDB=0 (default here) to disable W&B logging.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export FGNO_WANDB="${FGNO_WANDB:-0}"

DATA_DIR="data/dreamt/processed_BVP_ACC_up_sample_datasets"
FFM_CKPT="checkpoints/ffm_dreamt_BVP_ACC_upsample.pt"
MAE_CKPT="checkpoints/mae_dreamt_BVP_ACC_upsample_epoch_250.pt"

missing=0
for f in "$DATA_DIR/finetune_train_dataset_sub34.pt" \
         "$DATA_DIR/finetune_skin_train_dataset_sub34.pt" \
         "$FFM_CKPT" "$MAE_CKPT"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing: $f"
    missing=1
  fi
done
if [[ "$missing" -eq 1 ]]; then
  echo
  echo "One or more required inputs are missing -- see data_preprocess/dreamt/README.md" \
       "to produce the .pt splits, and pretrain/pretrain_ffm_dreamt.py /" \
       "pretrain/pretrain_mae_dreamt.py to produce the checkpoints."
  exit 1
fi

echo "FGNO root: $ROOT"

echo "========================================"
echo "[1/4] FFM sleep-stage classification (finetune_dreamt_clean_inference.py)"
echo "========================================"
python finetune/DREAMT_finetune/finetune_dreamt_clean_inference.py

echo "========================================"
echo "[2/4] FFM skin-temperature regression (finetune_dreamt_BVP_HR.py)"
echo "========================================"
python finetune/DREAMT_finetune/finetune_dreamt_BVP_HR.py

echo "========================================"
echo "[3/4] MAE sleep-stage classification (finetune_dreamt_mae.py)"
echo "========================================"
python finetune/DREAMT_finetune/finetune_dreamt_mae.py

echo "========================================"
echo "[4/4] MAE skin-temperature regression (finetune_dreamt_mae_skin_temp.py)"
echo "========================================"
python finetune/DREAMT_finetune/finetune_dreamt_mae_skin_temp.py

echo "All DREAMT fine-tuning experiments completed."
echo "Results: outputs/dreamt_sleep.csv, outputs/dreamt_skin.csv, outputs/dreamt_sleep_mae.csv, outputs/dreamt_skin_mae.csv"
