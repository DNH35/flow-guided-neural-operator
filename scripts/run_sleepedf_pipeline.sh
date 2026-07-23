#!/usr/bin/env bash
# End-to-end Sleep-EDF pipeline: prep data -> pretrain -> finetune -> results.
#
# Usage:
#   bash scripts/run_sleepedf_pipeline.sh /path/to/sleep-cassette
#
# The PhysioNet Sleep-EDF raw EDFs are not included in this repo (large,
# public download) -- pass the folder containing the downloaded
# *PSG.edf / *Hypnogram.edf pairs as the first argument. Skips a stage if
# its output already exists. Set FGNO_WANDB=0 to disable W&B logging (on
# by default). See data_preprocess/README.md for details.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

RAW_EDF_DIR="${1:-}"

if [[ ! -f data/sleepEDF_raw/train.pt ]]; then
  if [[ -z "$RAW_EDF_DIR" ]]; then
    echo "data/sleepEDF_raw is missing and no raw EDF directory was given."
    echo "Usage: bash scripts/run_sleepedf_pipeline.sh /path/to/sleep-cassette"
    exit 1
  fi
  echo "[1/5] Extracting labeled epochs from raw EDFs in $RAW_EDF_DIR ..."
  python data_preprocess/sleep_edf/preprocess_sleep_edf.py \
    --data_dir "$RAW_EDF_DIR" \
    --output_dir data_preprocess/sleep_edf/sleepEDF20_fpzcz \
    --subjects_output_dir data_preprocess/sleep_edf/sleepEDF20_fpzcz_subjects

  echo "[2/5] Combining per-subject epochs into data/sleepEDF_raw ..."
  python data_preprocess/sleep_edf/generate_train_val_test.py
else
  echo "[1-2/5] data/sleepEDF_raw already present, skipping."
fi

if [[ ! -f data/sleepEDF/train_stft.pt ]]; then
  echo "[3/5] STFT-transforming data/sleepEDF_raw -> data/sleepEDF ..."
  python data_preprocess/sleep_edf/preprocess_stft_sleepedf.py
else
  echo "[3/5] data/sleepEDF already present, skipping."
fi

if [[ ! -f checkpoints/ffm_sleepEDF_pretrain.pt ]]; then
  echo "[4/5] Pretraining FFM backbone on Sleep-EDF STFT data ..."
  python pretrain/pretrain_sleep_edf.py
else
  echo "[4/5] checkpoints/ffm_sleepEDF_pretrain.pt already present, skipping."
fi

echo "[5/5] Fine-tuning + evaluating (layer x extraction-time sweep) ..."
python finetune/finetune_sleepEDF.py

echo "Done. Results: outputs/sleepEDF_clean_results.csv"
