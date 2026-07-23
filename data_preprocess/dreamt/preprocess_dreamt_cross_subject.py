"""Cross-subject DREAMT BVP+ACC pipeline for FFM pretraining + sleep-stage finetuning.

Input:  raw 64Hz DREAMT participant CSVs (--data_dir), one file per subject,
        e.g. S002_whole_df.csv ... -- NOT included in this repo. Download the
        "data_64Hz" folder from PhysioNet's DREAMT 2.1.0 page.
Output: data/dreamt/processed_BVP_ACC_up_sample_datasets/
          pretrain_train_dataset.pt, pretrain_val_dataset.pt   (used by pretrain/pretrain_ffm_dreamt.py)
          finetune_train_dataset.pt, finetune_val_dataset.pt, finetune_test_dataset.pt

Each .pt file is a pickled PyTorch Dataset (not just tensors) -- consumers
must import from data.DREAMT_data.dataset.{pretrain_dataset,finetune_dataset}
(i.e. run with PYTHONPATH=<FGNO root>) for torch.load to unpickle them.
"""
import argparse
from pathlib import Path

import torch

from dreamt_dataloaders import get_multichannel_dataloaders

FGNO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_DIR = FGNO_ROOT / "data" / "dreamt" / "processed_BVP_ACC_up_sample_datasets"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Folder of raw 64Hz DREAMT participant CSVs (e.g. .../data_64Hz)")
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    loaders = get_multichannel_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)

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
