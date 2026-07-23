"""Combine per-subject Sleep-EDF npz files into train/val/test tensors.

Step 2 of 3 for the Sleep-EDF pipeline (see preprocess_sleep_edf.py).

Input:  sleepEDF20_fpzcz_subjects/subject_*.npz (from preprocess_sleep_edf.py)
Output: data/sleepEDF_raw/{train,val,test}.pt

Subjects are permuted with a fixed ordering before the 60/20/20 split so the
split matches the one used to produce the shipped data/sleepEDF_raw tensors
(and the original TS-TCC / AttnSleep paper results).
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

FGNO_ROOT = Path(__file__).resolve().parents[2]

# Fixed subject permutation so train/val/test splits match the paper / shipped data.
EDF20_PERMUTATION = np.array(
    [14, 5, 4, 17, 8, 7, 19, 12, 0, 15, 16, 9, 11, 10, 3, 1, 6, 18, 2, 13]
)


def load_split(files):
    X = np.load(files[0])["x"]
    y = np.load(files[0])["y"]
    for np_file in files[1:]:
        X = np.vstack((X, np.load(np_file)["x"]))
        y = np.append(y, np.load(np_file)["y"])
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "sleepEDF20_fpzcz_subjects"),
        help="Directory of per-subject .npz files produced by preprocess_sleep_edf.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(FGNO_ROOT / "data" / "sleepEDF_raw"),
        help="Where to write train.pt / val.pt / test.pt",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = os.listdir(args.data_dir)
    files = np.array([os.path.join(args.data_dir, i) for i in files])
    files.sort()
    files = files[EDF20_PERMUTATION]

    len_train = int(len(files) * 0.6)
    len_valid = int(len(files) * 0.2)

    splits = {
        "train": files[:len_train],
        "val": files[len_train:len_train + len_valid],
        "test": files[len_train + len_valid:],
    }

    for split, split_files in splits.items():
        X, y = load_split(split_files)
        data_save = {
            "samples": torch.from_numpy(X.transpose(0, 2, 1)),
            "labels": torch.from_numpy(y),
        }
        out_path = os.path.join(args.output_dir, f"{split}.pt")
        torch.save(data_save, out_path)
        print(f"Saved {out_path} -> samples {data_save['samples'].shape}, labels {data_save['labels'].shape}")


if __name__ == "__main__":
    main()
