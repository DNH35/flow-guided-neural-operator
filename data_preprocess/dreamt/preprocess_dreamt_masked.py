"""Cross-subject DREAMT BVP+ACC pipeline for masked-autoencoder (MAE) pretraining.

Same underlying pipeline as preprocess_dreamt_cross_subject.py, but each
pretrain window additionally carries a BERT-style block mask (see
data/DREAMT_data/dataset/pretrain_dataset.py: MaskingConfig, mask_inputs).

Input:  raw 64Hz DREAMT participant CSVs (--data_dir) -- see
        preprocess_dreamt_cross_subject.py for where to get them.
Output: data/dreamt/processed_BVP_ACC_masked_datasets/
          pretrain_train_dataset.pt, pretrain_val_dataset.pt   (used by pretrain/pretrain_mae_dreamt.py)
          finetune_train_dataset.pt, finetune_val_dataset.pt, finetune_test_dataset.pt
"""
import argparse
from pathlib import Path

import torch

from data.DREAMT_data.dataset.pretrain_dataset import MaskingConfig
from dreamt_dataloaders import get_multichannel_dataloaders

FGNO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_DIR = FGNO_ROOT / "data" / "dreamt" / "processed_BVP_ACC_masked_datasets"

MASKING_CONFIG = MaskingConfig(
    time_mask_p=0.1, time_mask_consecutive_min=3, time_mask_consecutive_max=7,
    freq_mask_p=0.1, freq_mask_consecutive_min=4, freq_mask_consecutive_max=8,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Folder of raw 64Hz DREAMT participant CSVs (e.g. .../data_64Hz)")
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    loaders = get_multichannel_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, masking_cfg=MASKING_CONFIG
    )

    for name, key in [
        ("pretrain_train_dataset.pt", "pretrain_train"),
        ("pretrain_val_dataset.pt", "pretrain_val"),
        ("finetune_train_dataset.pt", "finetune_train"),
        ("finetune_val_dataset.pt", "finetune_val"),
        ("finetune_test_dataset.pt", "finetune_test"),
    ]:
        out_path = save_dir / name
        torch.save(loaders[key].dataset, out_path)
        print(f"Saved {out_path} ({len(loaders[key].dataset)} windows)")


if __name__ == "__main__":
    main()
