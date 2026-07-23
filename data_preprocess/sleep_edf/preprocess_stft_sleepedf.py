"""STFT-transform the windowed Sleep-EDF tensors into the FFM input format.

Step 3 of 3 for the Sleep-EDF pipeline (see preprocess_sleep_edf.py).

Input:  data/sleepEDF_raw/{train,val,test}.pt   (produced by generate_train_val_test.py)
Output: data/sleepEDF/{train,val,test}_stft.pt
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from tqdm import tqdm

FGNO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = FGNO_ROOT / "data" / "sleepEDF_raw"
SAVE_DIR = FGNO_ROOT / "data" / "sleepEDF"

SAMPLING_RATE = 100
STFT_CONFIG = {
    "freq_channel_cutoff": -1,
    "nperseg": 75,
    "noverlap": 70,
    "normalizing": "zscore",
}


def zscore(a, axis):
    mn = a.mean(axis=axis, keepdims=True)
    std = a.std(axis=axis, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    return (a - mn) / std


class STFTPreprocessor(nn.Module):
    def __init__(self, cfg, sampling_rate):
        super().__init__()
        self.cfg = cfg
        self.sampling_rate = sampling_rate

    def get_stft(self, x, fs, show_fs=-1, normalizing=None, **kwargs):
        if x.ndim > 1:
            x = x.squeeze()

        f, t, Zxx = signal.stft(x, fs, **kwargs)

        if kwargs.get("return_onesided") and show_fs != -1:
            Zxx = Zxx[:show_fs]

        if normalizing == "zscore":
            Zxx = zscore(Zxx, axis=-1)
            if (Zxx.std() == 0).any():
                Zxx = np.ones_like(Zxx)
        elif normalizing == "db":
            Zxx = np.log(Zxx + 1e-10)

        if np.isnan(Zxx).any():
            Zxx = np.nan_to_num(Zxx, nan=0.0)

        return torch.Tensor(Zxx)

    def forward(self, wav: np.ndarray):
        all_channels_stft = []
        for i in range(wav.shape[0]):
            channel_signal = wav[i, :]
            stft_tensor = self.get_stft(
                channel_signal,
                self.sampling_rate,
                show_fs=self.cfg["freq_channel_cutoff"],
                nperseg=self.cfg["nperseg"],
                noverlap=self.cfg["noverlap"],
                normalizing=self.cfg["normalizing"],
                return_onesided=True,
            )
            all_channels_stft.append(stft_tensor)
        return torch.cat(all_channels_stft, dim=0)


def preprocess_and_save(data_dir, save_dir, cfg, sampling_rate):
    save_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = STFTPreprocessor(cfg, sampling_rate)

    for split in ["train", "val", "test"]:
        input_path = data_dir / f"{split}.pt"
        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping.")
            continue

        data = torch.load(input_path)
        samples, labels = data["samples"], data["labels"]

        stft_samples = []
        for i in tqdm(range(len(samples)), desc=f"Converting {split} samples"):
            x = samples[i].numpy()
            stft_samples.append(preprocessor(x))

        stft_samples_tensor = torch.stack(stft_samples)
        output_path = save_dir / f"{split}_stft.pt"
        torch.save({"samples": stft_samples_tensor, "labels": labels}, output_path)
        print(f"Saved {output_path} -> {stft_samples_tensor.shape} (N, FreqBins, TimeSteps)")


if __name__ == "__main__":
    preprocess_and_save(DATA_DIR, SAVE_DIR, STFT_CONFIG, SAMPLING_RATE)
