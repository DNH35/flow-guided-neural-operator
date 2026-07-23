"""Unsupervised (pretraining) Dataset for the DREAMT BVP+ACC pipeline.

PretrainDataset optionally applies block masking (BERT-style 80/10/10) for
MAE-style pretraining -- pass masking_cfg=None (the default) for plain FFM
pretraining (__getitem__ returns {'target': stft}), or a MaskingConfig for
masked-autoencoder pretraining (__getitem__ additionally returns
'masked_input' and 'mask_label').

Kept as an importable module (not defined inline in a notebook or in
pretrain/pretrain_ffm_dreamt.py / pretrain/pretrain_mae_dreamt.py) so that
torch.load(...) can unpickle saved PretrainDataset instances from any script,
as long as it runs with the FGNO root on PYTHONPATH.
"""
from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset


class MaskingConfig:
    def __init__(
        self,
        time_mask_p: float,
        time_mask_consecutive_min: int,
        time_mask_consecutive_max: int,
        freq_mask_p: float,
        freq_mask_consecutive_min: int,
        freq_mask_consecutive_max: int,
    ):
        self.time_mask_p = time_mask_p
        self.time_mask_consecutive_min = time_mask_consecutive_min
        self.time_mask_consecutive_max = time_mask_consecutive_max
        self.freq_mask_p = freq_mask_p
        self.freq_mask_consecutive_min = freq_mask_consecutive_min
        self.freq_mask_consecutive_max = freq_mask_consecutive_max


def mask_inputs(data: torch.Tensor, cfg: MaskingConfig):
    """Block-masks a (time, freq) spectrogram with BERT-style 80/10/10 replacement.

    Returns (masked_data, mask_label) where mask_label is a same-shape float
    tensor that is 1.0 wherever a value was selected for masking.
    """
    masked_data = data.clone()
    mask_label = torch.zeros_like(data, dtype=torch.bool)
    mask_fill_value = 0.0

    seq_len_time = data.shape[0]
    time_intervals = []
    for i in range(seq_len_time):
        if random.random() < cfg.time_mask_p:
            if not time_intervals or i >= time_intervals[-1][1]:
                mask_len = random.randint(cfg.time_mask_consecutive_min, cfg.time_mask_consecutive_max)
                time_intervals.append((i, min(i + mask_len, seq_len_time)))

    for start, end in time_intervals:
        mask_label[start:end, :] = True
        dice = random.random()
        patch_len = end - start
        if dice < 0.8:
            masked_data[start:end, :] = mask_fill_value
        elif dice < 0.9:
            rand_start = random.randint(0, seq_len_time - patch_len)
            masked_data[start:end, :] = data[rand_start:rand_start + patch_len, :]

    seq_len_freq = data.shape[1]
    freq_intervals = []
    for i in range(seq_len_freq):
        if random.random() < cfg.freq_mask_p:
            if not freq_intervals or i >= freq_intervals[-1][1]:
                mask_len = random.randint(cfg.freq_mask_consecutive_min, cfg.freq_mask_consecutive_max)
                freq_intervals.append((i, min(i + mask_len, seq_len_freq)))

    for start, end in freq_intervals:
        mask_label[:, start:end] = True
        dice = random.random()
        patch_len = end - start
        if dice < 0.8:
            masked_data[:, start:end] = mask_fill_value
        elif dice < 0.9:
            rand_start = random.randint(0, seq_len_freq - patch_len)
            masked_data[:, start:end] = data[:, rand_start:rand_start + patch_len]

    return masked_data, mask_label.float()


class PretrainDataset(Dataset):
    def __init__(self, data_windows, preprocessor, masking_cfg: MaskingConfig = None):
        self.data_windows = data_windows
        self.preprocessor = preprocessor
        self.masking_cfg = masking_cfg

    def __len__(self):
        return len(self.data_windows)

    def __getitem__(self, idx):
        target = self.preprocessor(self.data_windows[idx])
        if self.masking_cfg is not None:
            masked_input, mask_label = mask_inputs(target, self.masking_cfg)
            return {"masked_input": masked_input, "mask_label": mask_label, "target": target}
        return {"target": target}
