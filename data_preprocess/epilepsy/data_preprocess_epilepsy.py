"""Turn the raw UCI Epilepsy CSV export into train/val/test tensors.

Input:  data_preprocess/epilepsy/epilepsy_srcdata.csv
Output: data/epilepsy_raw/{train,val,test}.pt

Each row of the CSV is a single 178-sample window (1 channel). Labels are
recoded to binary: 0 = seizure (original class 1), 1 = non-seizure (original
classes 2-5). Matches the split used to produce the shipped data/epilepsy_raw
tensors (fixed random_state=42, 64/16/20 train/val/test).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

FGNO_ROOT = Path(__file__).resolve().parents[2]
SRC_CSV = Path(__file__).resolve().parent / "epilepsy_srcdata.csv"
OUTPUT_DIR = FGNO_ROOT / "data" / "epilepsy_raw"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(SRC_CSV)

    y = data.iloc[:, -1].to_numpy()
    x = data.iloc[:, 1:-1].to_numpy()

    y = y - 1
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    # Binary relabel: keep class 0 (seizure) as 0, collapse all other
    # classes into 1 (non-seizure).
    for i, j in enumerate(y):
        if j != 0:
            y[i] = 1

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    for split, (X, Y) in {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }.items():
        dat_dict = {
            "samples": torch.from_numpy(X).unsqueeze(1),
            "labels": torch.from_numpy(Y),
        }
        out_path = OUTPUT_DIR / f"{split}.pt"
        torch.save(dat_dict, out_path)
        print(f"Saved {out_path} -> samples {dat_dict['samples'].shape}, labels {dat_dict['labels'].shape}")


if __name__ == "__main__":
    main()
