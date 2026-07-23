"""Builds DREAMT BVP+ACC train/val/test DataLoaders from raw 64Hz participant CSVs.

Shared by preprocess_dreamt_cross_subject.py, preprocess_dreamt_masked.py, and
preprocess_dreamt_subject_splits.py. Ported from DREAMT_FE's
bvp_datapreprocessing.ipynb / bvp_masked_datapreprocessing.ipynb.

Fixed vs. the source notebooks: get_multichannel_dataloaders there hardcoded
`train_files, val_files, test_files = participant_files[:1], [1:2], [2:3]`
regardless of `split_ratios` or how many participants were available --
i.e. it silently only ever used 3 participants total. That looks like a
debug shortcut left in place rather than the intended behavior (the
`split_ratios` parameter and the surrounding quality-control code make no
sense otherwise), so this version restores a real participant-level
shuffle+split across every quality-passing participant, using
`load_high_quality_participant_files`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.DREAMT_data.dataset.finetune_dataset import FinetuneDataset
from data.DREAMT_data.dataset.pretrain_dataset import MaskingConfig, PretrainDataset
from data.DREAMT_data.dataset.preprocessing import (
    EPOCH_SIZE,
    NOVERLAP,
    NPERSEG,
    SAMPLING_RATE,
    WINDOW_SIZE,
    STFTPreprocessor,
    is_epoch_artifact,
    load_high_quality_participant_files,
    preprocess_ACC_for_ssl,
    preprocess_BVP_for_ssl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VALID_SLEEP_STAGES = ["P", "N1", "N2", "N3", "R", "W"]


def _stft_preprocessor():
    cfg = SimpleNamespace(nperseg=NPERSEG, noverlap=NOVERLAP, normalizing="zscore", freq_channel_cutoff=-1)
    return STFTPreprocessor(cfg=cfg, sampling_rate=SAMPLING_RATE)


def _windows_and_labels_for_file(csv_path):
    """Returns (windows: list[np.ndarray[WINDOW_SIZE, 4]], labels: list[int]) for one participant."""
    df = pd.read_csv(csv_path)
    try:
        first_wake_index = df[df["Sleep_Stage"] == "W"].index[0]
        start_index = first_wake_index - (first_wake_index % EPOCH_SIZE)
        df = df.iloc[start_index:].copy()
    except IndexError:
        logging.warning(f"No 'W' stage in {csv_path.name}. Skipping file.")
        return [], []

    df.reset_index(drop=True, inplace=True)
    df.ffill(inplace=True)
    df["binary_label"] = df["Sleep_Stage"].apply(lambda x: 0 if x == "W" else 1)
    valid_stages_mask = df["Sleep_Stage"].isin(VALID_SLEEP_STAGES)

    multichannel_signal = np.stack([
        preprocess_BVP_for_ssl(df["BVP"].to_numpy()),
        preprocess_ACC_for_ssl(df["ACC_X"].to_numpy()),
        preprocess_ACC_for_ssl(df["ACC_Y"].to_numpy()),
        preprocess_ACC_for_ssl(df["ACC_Z"].to_numpy()),
    ], axis=1)
    labels_full = df["binary_label"].to_numpy()

    windows, labels = [], []
    for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        if valid_stages_mask[i:i + WINDOW_SIZE].all():
            window = multichannel_signal[i:i + WINDOW_SIZE]
            if window.shape[0] == WINDOW_SIZE:
                windows.append(window)
                labels.append(labels_full[i])
    return windows, labels


def get_multichannel_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    split_ratios: tuple = (0.8, 0.1, 0.1),
    masking_cfg: MaskingConfig = None,
    run_quality_control: bool = True,
):
    """Cross-subject BVP+ACC pipeline: QC -> participant split -> windowing -> SMOTE (train only).

    Returns a dict of DataLoaders: pretrain_train, pretrain_val, finetune_train,
    finetune_val, finetune_test.
    """
    preprocessor = _stft_preprocessor()

    if run_quality_control:
        participant_files = load_high_quality_participant_files(data_dir)
    else:
        participant_files = sorted(Path(data_dir).glob("*.csv"))
        if not participant_files:
            raise FileNotFoundError(f"No CSV files found in '{data_dir}'.")

    rng = np.random.default_rng(42)
    participant_files = list(participant_files)
    rng.shuffle(participant_files)
    n_train = int(len(participant_files) * split_ratios[0])
    n_val = int(len(participant_files) * split_ratios[1])
    train_files = participant_files[:n_train]
    val_files = participant_files[n_train:n_train + n_val]
    test_files = participant_files[n_train + n_val:]
    logging.info(f"Splitting {len(participant_files)} participants into "
                 f"{len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")

    all_windows = {"train": [], "val": [], "test": []}
    all_labels = {"train": [], "val": [], "test": []}
    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        for f in files:
            windows, labels = _windows_and_labels_for_file(f)
            all_windows[split_name].extend(windows)
            all_labels[split_name].extend(labels)

    X_train_original = np.array(all_windows["train"])
    y_train_original = np.array(all_labels["train"])
    unique, counts = np.unique(y_train_original, return_counts=True)
    logging.info(f"Original training set distribution: {dict(zip(unique, counts))}")

    if len(unique) > 1:
        n_samples, n_timesteps, n_features = X_train_original.shape
        smote = SMOTE(random_state=42)
        X_resampled_flat, y_train_resampled = smote.fit_resample(
            X_train_original.reshape(n_samples, -1), y_train_original
        )
        X_train_resampled = X_resampled_flat.reshape(-1, n_timesteps, n_features)
    else:
        logging.warning("Only one class found in training data. Skipping SMOTE.")
        X_train_resampled, y_train_resampled = X_train_original, y_train_original

    pretrain_train_dataset = PretrainDataset(X_train_resampled, preprocessor, masking_cfg=masking_cfg)
    pretrain_val_dataset = PretrainDataset(np.array(all_windows["val"]), preprocessor, masking_cfg=masking_cfg)
    finetune_train_dataset = FinetuneDataset(X_train_resampled, y_train_resampled, preprocessor)
    finetune_val_dataset = FinetuneDataset(np.array(all_windows["val"]), np.array(all_labels["val"]), preprocessor)
    finetune_test_dataset = FinetuneDataset(np.array(all_windows["test"]), np.array(all_labels["test"]), preprocessor)

    return {
        "pretrain_train": DataLoader(pretrain_train_dataset, batch_size=batch_size, shuffle=True),
        "pretrain_val": DataLoader(pretrain_val_dataset, batch_size=batch_size, shuffle=False),
        "finetune_train": DataLoader(finetune_train_dataset, batch_size=batch_size, shuffle=True),
        "finetune_val": DataLoader(finetune_val_dataset, batch_size=batch_size, shuffle=False),
        "finetune_test": DataLoader(finetune_test_dataset, batch_size=batch_size, shuffle=False),
    }


def get_multichannel_subject_split_dataloaders(
    subject_id: str,
    data_dir: str,
    batch_size: int = 32,
    split_ratios: tuple = (0.8, 0.1, 0.1),
    use_smote: bool = True,
):
    """Single-subject BVP+ACC binary sleep-stage (Wake vs. not-Wake) classification split."""
    logging.info(f"Creating multi-channel train/val/test splits for single subject: {subject_id}")
    subject_file = Path(data_dir) / subject_id
    if not subject_file.exists():
        raise FileNotFoundError(f"Subject file not found: {subject_file}")

    df = pd.read_csv(subject_file)
    try:
        first_wake_index = df[df["Sleep_Stage"] == "W"].index[0]
        start_index = first_wake_index - (first_wake_index % EPOCH_SIZE)
        trimmed_df = df.iloc[start_index:].copy()
    except IndexError:
        raise ValueError(f"Cannot perform QC: subject {subject_id} has no 'W' stage.")

    total_epoch_count = artifact_epoch_count = 0
    for i in range(0, len(trimmed_df), EPOCH_SIZE):
        epoch_df = trimmed_df.iloc[i:i + EPOCH_SIZE]
        if len(epoch_df) < EPOCH_SIZE:
            continue
        total_epoch_count += 1
        if is_epoch_artifact(epoch_df):
            artifact_epoch_count += 1
    if total_epoch_count == 0:
        raise ValueError(f"Subject {subject_id} has no full epochs post-trimming.")
    artifact_percentage = (artifact_epoch_count / total_epoch_count) * 100
    logging.info(f"Subject {subject_id}: {artifact_percentage:.2f}% artifacts "
                 f"({'PASSED' if artifact_percentage <= 20.0 else 'FAILED'} QC).")

    preprocessor = _stft_preprocessor()
    windows, labels = _windows_and_labels_for_file(subject_file)
    X, y = np.array(windows), np.array(labels)
    logging.info(f"Extracted {len(X)} total multi-channel windows from subject {subject_id}.")

    train_ratio, val_ratio, test_ratio = split_ratios
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), random_state=42, stratify=y
    )
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_test_ratio, random_state=42, stratify=y_temp
    )
    logging.info(f"Data split into: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test windows.")

    if use_smote and len(np.unique(y_train)) > 1:
        n_samples, n_timesteps, n_features = X_train.shape
        smote = SMOTE(random_state=42)
        X_resampled_flat, y_train = smote.fit_resample(X_train.reshape(n_samples, -1), y_train)
        X_train = X_resampled_flat.reshape(-1, n_timesteps, n_features)

    return {
        "train": DataLoader(FinetuneDataset(X_train, y_train, preprocessor), batch_size=batch_size, shuffle=True),
        "val": DataLoader(FinetuneDataset(X_val, y_val, preprocessor), batch_size=batch_size, shuffle=False),
        "test": DataLoader(FinetuneDataset(X_test, y_test, preprocessor), batch_size=batch_size, shuffle=False),
    }


def get_multichannel_subject_temp_sequence_dataloaders(
    subject_id: str,
    data_dir: str,
    batch_size: int = 32,
):
    """Single-subject BVP+ACC -> skin-temperature sequence regression split.

    Each 5s/320-sample window is paired with a 20-sample (4Hz) TEMP sequence label.
    """
    logging.info(f"Creating multi-channel -> TEMP sequence dataloaders for subject: {subject_id}")
    preprocessor = _stft_preprocessor()

    subject_file = Path(data_dir) / subject_id
    if not subject_file.exists():
        raise FileNotFoundError(f"Subject file not found: {subject_file}")

    df = pd.read_csv(subject_file)
    try:
        first_wake_index = df[df["Sleep_Stage"] == "W"].index[0]
        start_index = first_wake_index - (first_wake_index % EPOCH_SIZE)
        df = df.iloc[start_index:].copy()
    except IndexError:
        logging.warning(f"No 'W' stage in {subject_id}, using file from the beginning.")

    df.reset_index(drop=True, inplace=True)
    df["TEMP"] = pd.to_numeric(df["TEMP"], errors="coerce")
    df.ffill(inplace=True)
    df.dropna(subset=["TEMP", "BVP", "ACC_X", "ACC_Y", "ACC_Z"], inplace=True)

    multichannel_signal = np.stack([
        preprocess_BVP_for_ssl(df["BVP"].to_numpy()),
        preprocess_ACC_for_ssl(df["ACC_X"].to_numpy()),
        preprocess_ACC_for_ssl(df["ACC_Y"].to_numpy()),
        preprocess_ACC_for_ssl(df["ACC_Z"].to_numpy()),
    ], axis=1)
    temp_labels_64hz = df["TEMP"].to_numpy()

    windows, labels = [], []
    for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        input_window = multichannel_signal[i:i + WINDOW_SIZE]
        label_sequence_4hz = temp_labels_64hz[i:i + WINDOW_SIZE][::16]  # 320 -> 20 samples
        if input_window.shape[0] == WINDOW_SIZE and label_sequence_4hz.shape[0] == 20:
            windows.append(input_window)
            labels.append(label_sequence_4hz)

    X, y = np.array(windows), np.array(labels)
    logging.info(f"Extracted {len(X)} windows. Label length per window: {y.shape[1] if len(y) else 0}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    logging.info(f"Data split into: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test.")

    return {
        "train": DataLoader(FinetuneDataset(X_train, y_train, preprocessor), batch_size=batch_size, shuffle=True),
        "val": DataLoader(FinetuneDataset(X_val, y_val, preprocessor), batch_size=batch_size, shuffle=False),
        "test": DataLoader(FinetuneDataset(X_test, y_test, preprocessor), batch_size=batch_size, shuffle=False),
    }
