"""Single-subject DREAMT finetuning splits (sleep-stage classification + skin-temp regression).

Input:  raw 64Hz DREAMT participant CSVs (--data_dir) -- see
        preprocess_dreamt_cross_subject.py for where to get them.
Output (both written into the up-sample data dir, matching conf/dreamt_*.yaml):
  data/dreamt/processed_BVP_ACC_up_sample_datasets/
    finetune_{train,val,test}_dataset_sub<ID>.pt        (conf/dreamt_sleep_finetune.yaml / dreamt_sleep_mae_finetune.yaml)
    finetune_skin_{train,val,test}_dataset_sub<ID>.pt    (conf/dreamt_skin_finetune.yaml / dreamt_skin_mae_finetune.yaml)

Defaults to subject S034 (--subject), matching the shipped configs' "sub34" filenames.
"""
import argparse
from pathlib import Path

import torch

from dreamt_dataloaders import (
    get_multichannel_subject_split_dataloaders,
    get_multichannel_subject_temp_sequence_dataloaders,
)

FGNO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_DIR = FGNO_ROOT / "data" / "dreamt" / "processed_BVP_ACC_up_sample_datasets"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Folder of raw 64Hz DREAMT participant CSVs (e.g. .../data_64Hz)")
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--subject", type=str, default="S034_whole_df.csv")
    parser.add_argument("--suffix", type=str, default="sub34", help="Filename suffix, e.g. 'sub34'")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Sleep-stage classification split for {args.subject} ---")
    sleep_loaders = get_multichannel_subject_split_dataloaders(
        subject_id=args.subject, data_dir=args.data_dir, batch_size=32, use_smote=True
    )
    for split, loader in sleep_loaders.items():
        out_path = save_dir / f"finetune_{split}_dataset_{args.suffix}.pt"
        torch.save(loader, out_path)
        print(f"Saved {out_path} ({len(loader.dataset)} windows)")

    print(f"\n--- Skin-temperature regression split for {args.subject} ---")
    skin_loaders = get_multichannel_subject_temp_sequence_dataloaders(
        subject_id=args.subject, data_dir=args.data_dir, batch_size=128
    )
    for split, loader in skin_loaders.items():
        out_path = save_dir / f"finetune_skin_{split}_dataset_{args.suffix}.pt"
        torch.save(loader, out_path)
        print(f"Saved {out_path} ({len(loader.dataset)} windows)")


if __name__ == "__main__":
    main()
