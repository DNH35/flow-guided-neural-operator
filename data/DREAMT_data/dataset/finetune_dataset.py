"""Supervised (finetuning) Dataset for the DREAMT BVP+ACC pipeline.

Labels are stored as-is: a scalar per window for the binary sleep-stage
classification task, or a length-20 sequence per window for the skin
temperature regression task. Kept as an importable module for the same
pickling reason as pretrain_dataset.py (see its docstring).
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class FinetuneDataset(Dataset):
    def __init__(self, data_windows, labels, preprocessor):
        self.data_windows = data_windows
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.data_windows)

    def __getitem__(self, idx):
        stft_tensor = self.preprocessor(self.data_windows[idx])
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {"input": stft_tensor, "labels": label_tensor}
