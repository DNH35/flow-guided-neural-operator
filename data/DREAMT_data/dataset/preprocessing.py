"""Shared signal-processing building blocks for the DREAMT (BVP+ACC) pipeline.

Ported from the multi-channel pipeline in DREAMT_FE's bvp_datapreprocessing.ipynb /
bvp_masked_datapreprocessing.ipynb. Kept in one importable module (rather than
copy-pasted into every consumer script) because instances of STFTPreprocessor
get pickled inside the saved PretrainDataset/FinetuneDataset objects
(see pretrain_dataset.py / finetune_dataset.py) -- torch.load needs this exact
module path importable wherever those .pt files are later loaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

SAMPLING_RATE = 64  # Hz, unified sampling rate for BVP + ACC
WINDOW_SECONDS = 5
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS  # 320 samples/window
EPOCH_SIZE = SAMPLING_RATE * 30  # 1920 samples, for alignment with 30s sleep-stage labels
NUM_CHANNELS = 4  # BVP, ACC_X, ACC_Y, ACC_Z

NPERSEG = 64
NOVERLAP = 48
ARTIFACT_THRESHOLD = 20.0  # percent


def zscore(a, axis):
    mn = a.mean(axis=axis, keepdims=True)
    std = a.std(axis=axis, ddof=0, keepdims=True)
    std[(std == 0)] = 1.0
    return (a - mn) / std


def preprocess_BVP_for_ssl(bvp_64hz: np.ndarray) -> np.ndarray:
    """Chebyshev-II bandpass (0.5-20Hz) + z-score normalize."""
    lowcut, highcut, order = 0.5, 20.0, 5
    sos = signal.cheby2(N=order, rs=30, Wn=[lowcut, highcut], btype="bandpass", fs=SAMPLING_RATE, output="sos")
    bvp_filtered = signal.sosfilt(sos, bvp_64hz)
    return (bvp_filtered - np.mean(bvp_filtered)) / (np.std(bvp_filtered) + 1e-8)


def preprocess_ACC_for_ssl(acc_64hz_raw: np.ndarray) -> np.ndarray:
    """Downsample to 32Hz, bandpass 3-10Hz, upsample back to 64Hz, z-score normalize."""
    acc_32hz = acc_64hz_raw[::2]
    sos = signal.butter(N=3, Wn=[3, 10], btype="bp", fs=32, output="sos")
    acc_32hz_filtered = signal.sosfilt(sos, acc_32hz)

    original_len, target_len = len(acc_32hz_filtered), len(acc_64hz_raw)
    original_time = np.linspace(0, original_len - 1, original_len)
    target_time = np.linspace(0, original_len - 1, target_len)
    interpolator = interp1d(original_time, acc_32hz_filtered, kind="linear", fill_value="extrapolate")
    acc_64hz_interpolated = interpolator(target_time)

    return (acc_64hz_interpolated - np.mean(acc_64hz_interpolated)) / (np.std(acc_64hz_interpolated) + 1e-8)


def is_epoch_artifact(epoch_df: pd.DataFrame) -> bool:
    """True if a 30s epoch looks like a BVP/ACC artifact (low SNR, clipping, high motion)."""
    bvp = epoch_df.BVP.to_numpy()
    b, a = butter(N=2, Wn=[0.5 / (0.5 * SAMPLING_RATE), 15 / (0.5 * SAMPLING_RATE)], btype="band")
    filtered_signal = filtfilt(b, a, bvp)
    signal_power = np.mean(filtered_signal ** 2)
    noise_power = np.mean((bvp - filtered_signal) ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    acc_x = (epoch_df.ACC_X.to_numpy() / 64)[::2]
    acc_y = (epoch_df.ACC_Y.to_numpy() / 64)[::2]
    acc_z = (epoch_df.ACC_Z.to_numpy() / 64)[::2]
    acc_std = np.std(np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2))

    return acc_std >= (0.4125 / 2) or snr_db < 10 or np.max(bvp) > 500 or np.min(bvp) < -500


class STFTPreprocessor(nn.Module):
    """Computes STFT for each channel of a multi-channel signal and concatenates
    the results along the feature dimension.

    Input: wav of shape (window_size, num_channels). Output: (TimeSteps, FreqBins * NumChannels).
    """

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
            f = f[:show_fs]
        Zxx = np.abs(Zxx)
        if normalizing == "zscore":
            Zxx = zscore(Zxx, axis=-1)
            if (Zxx.std() == 0).any():
                Zxx = np.ones_like(Zxx)
        elif normalizing == "db":
            Zxx = np.log(Zxx + 1e-10)
        if np.isnan(Zxx).any():
            Zxx = np.nan_to_num(Zxx, nan=0.0)
        return f, t, torch.Tensor(np.transpose(Zxx))

    def forward(self, wav: np.ndarray):
        all_channels_stft = []
        for i in range(wav.shape[-1]):
            channel_signal = wav[:, i]
            _, _, linear = self.get_stft(
                channel_signal, self.sampling_rate, show_fs=self.cfg.freq_channel_cutoff,
                nperseg=self.cfg.nperseg, noverlap=self.cfg.noverlap,
                normalizing=self.cfg.normalizing, return_onesided=True,
            )
            all_channels_stft.append(linear)
        return torch.cat(all_channels_stft, dim=1)


def load_high_quality_participant_files(data_dir, epoch_size=EPOCH_SIZE, artifact_threshold=ARTIFACT_THRESHOLD):
    """Quality-control pass: drop participant CSVs with no 'W' stage or a high
    BVP/ACC artifact rate. Returns the filtered, sorted list of file paths.

    Ported from the (commented-out in the source notebooks, but functioning)
    QC step in get_multichannel_dataloaders -- the shipped pipeline had this
    disabled in favor of a `participant_files[:1]` debug shortcut that only
    ever used 3 participants total regardless of dataset size. We restore the
    QC + full-dataset split here since that shortcut looked like leftover
    debug code, not the intended behavior.
    """
    from pathlib import Path
    from tqdm import tqdm
    import logging

    participant_files = sorted(Path(data_dir).glob("*.csv"))
    if not participant_files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'.")

    high_quality_files = []
    for f in tqdm(participant_files, desc="Checking participant data quality"):
        df = pd.read_csv(f)
        try:
            first_wake_index = df[df["Sleep_Stage"] == "W"].index[0]
            start_index = first_wake_index - (first_wake_index % epoch_size)
            trimmed_df = df.iloc[start_index:].copy()
        except IndexError:
            logging.warning(f"Participant {f.name} has no 'W' stage. Excluding.")
            continue

        total_epoch_count = artifact_epoch_count = 0
        for i in range(0, len(trimmed_df), epoch_size):
            epoch_df = trimmed_df.iloc[i:i + epoch_size]
            if len(epoch_df) < epoch_size:
                continue
            total_epoch_count += 1
            if is_epoch_artifact(epoch_df):
                artifact_epoch_count += 1

        if total_epoch_count == 0:
            logging.warning(f"Participant {f.name} has no full epochs post-trimming. Excluding.")
            continue

        artifact_percentage = (artifact_epoch_count / total_epoch_count) * 100
        if artifact_percentage > artifact_threshold:
            logging.warning(f"Excluding {f.name}: {artifact_percentage:.2f}% artifacts.")
        else:
            high_quality_files.append(f)

    logging.info(f"QC complete: {len(high_quality_files)}/{len(participant_files)} participants retained.")
    return high_quality_files
