"""Chronos embedding baseline on Brain Treebank speech (downsample-factor sweep).

Public-release experiment corresponding to Chronos results reported alongside FGNO.
Does not modify the FGNO FFM fine-tuning pipeline.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_FGNO_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_FGNO_ROOT))

import tasks
from data.BBT_data.datasets.chronos_wav_dataset import WavsChronosEmbeddingDataset

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.linear_out = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        return self.linear_out(x)


def resolve_path(path: str | Path, base: Path = _FGNO_ROOT) -> Path:
    path = Path(os.path.expanduser(str(path)))
    return path if path.is_absolute() else base / path


def load_cfg(config_name: str = "chronos_bbt_downsample.yaml"):
    cfg_path = _FGNO_ROOT / "conf" / config_name
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    cfg.data.raw_brain_data_dir = str(resolve_path(cfg.data.raw_brain_data_dir))
    cfg.results.save_path = str(resolve_path(cfg.results.save_path))
    return cfg


def extract_data_from_loader(loader):
    all_wavs, all_labels = [], []
    for batch in tqdm(loader, desc="Collecting waveforms"):
        all_wavs.extend(batch["wavs"])
        all_labels.extend(batch["labels"])
    return all_wavs, all_labels


def test_model(model, test_loader, device):
    model.eval()
    test_labels, test_predicts_proba = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).to(device)
            outputs = model(inputs)
            test_labels.extend(labels.cpu().numpy().flatten())
            test_predicts_proba.extend(torch.sigmoid(outputs).cpu().numpy().flatten())

    if len(np.unique(test_labels)) <= 1:
        print("Test set contains only one class. Cannot compute AUROC.")
        return None

    return {
        "accuracy": accuracy_score(test_labels, np.round(test_predicts_proba)),
        "auroc": roc_auc_score(test_labels, test_predicts_proba),
    }


def train_one_factor(
    train_loader,
    val_loader,
    input_dim: int,
    device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
):
    model = ClassifierHead(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auroc = 0.0
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_labels_data, train_predicts = [], []

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_labels_data.extend(labels.cpu().numpy().flatten())
            train_predicts.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        train_auroc = (
            roc_auc_score(train_labels_data, train_predicts)
            if len(np.unique(train_labels_data)) > 1
            else 0.0
        )

        model.eval()
        total_val_loss = 0.0
        val_labels_data, val_predicts = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.unsqueeze(1).to(device)
                outputs = model(inputs)
                total_val_loss += criterion(outputs, labels).item()
                val_labels_data.extend(labels.cpu().numpy().flatten())
                val_predicts.extend(torch.sigmoid(outputs).cpu().numpy().flatten())

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        val_auroc = (
            roc_auc_score(val_labels_data, val_predicts)
            if len(np.unique(val_labels_data)) > 1
            else 0.0
        )
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Train AUROC: {train_auroc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val AUROC: {val_auroc:.4f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"New best validation AUROC: {best_val_auroc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def main(config_name: str = "chronos_bbt_downsample.yaml"):
    try:
        from chronos import ChronosPipeline
    except ImportError as exc:
        raise ImportError(
            "Chronos is required for this experiment. Install with:\n"
            "  pip install chronos-forecasting\n"
            "or see requirements-chronos.txt"
        ) from exc

    cfg = load_cfg(config_name)
    logging.info("Chronos BBT downsample experiment")
    logging.info("Working directory: %s", os.getcwd())

    task = tasks.setup_task(cfg.exp.task)
    task.load_datasets(cfg.data, cfg.data.preprocessor)

    def get_batch_iterator(dataset, batch_size, **kwargs):
        return task.get_batch_iterator(dataset, batch_size, **kwargs)

    train_loader_original = get_batch_iterator(
        task.train_set,
        cfg.exp.runner.train_batch_size,
        shuffle=cfg.exp.runner.shuffle,
        num_workers=cfg.exp.runner.num_workers,
        persistent_workers=cfg.exp.runner.num_workers > 0,
    )
    valid_loader_original = get_batch_iterator(
        task.valid_set,
        cfg.exp.runner.valid_batch_size,
        shuffle=cfg.exp.runner.shuffle,
        num_workers=cfg.exp.runner.num_workers,
        persistent_workers=cfg.exp.runner.num_workers > 0,
    )
    test_loader_original = get_batch_iterator(
        task.test_set,
        cfg.exp.runner.valid_batch_size,
        shuffle=cfg.exp.runner.shuffle,
        num_workers=cfg.exp.runner.num_workers,
        persistent_workers=cfg.exp.runner.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chronos_cfg = cfg.chronos
    batch_size = int(chronos_cfg.batch_size)
    num_workers = int(chronos_cfg.num_workers)
    downsample_factors = list(chronos_cfg.downsample_factors)

    pipeline = ChronosPipeline.from_pretrained(
        chronos_cfg.model_id,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    train_wavs, train_labels = extract_data_from_loader(train_loader_original)
    val_wavs, val_labels = extract_data_from_loader(valid_loader_original)
    test_wavs, test_labels = extract_data_from_loader(test_loader_original)

    results = []
    results_path = Path(cfg.results.save_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    for factor in downsample_factors:
        print(f"\n=== Downsample factor {factor} ===")
        train_dataset = WavsChronosEmbeddingDataset(
            train_wavs, train_labels, pipeline, downsample_factor=factor
        )
        val_dataset = WavsChronosEmbeddingDataset(
            val_wavs, val_labels, pipeline, downsample_factor=factor
        )
        test_dataset = WavsChronosEmbeddingDataset(
            test_wavs, test_labels, pipeline, downsample_factor=factor
        )

        with torch.no_grad():
            first_embedding, _ = train_dataset[0]
            input_dim = int(first_embedding.shape[0])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        model = train_one_factor(
            train_loader,
            val_loader,
            input_dim=input_dim,
            device=device,
            epochs=int(chronos_cfg.epochs),
            patience=int(chronos_cfg.patience),
            lr=float(chronos_cfg.lr),
            weight_decay=float(chronos_cfg.weight_decay),
        )

        if len(test_loader.dataset) == 0:
            print("Test set is empty. Skipping evaluation.")
            continue

        test_results = test_model(model, test_loader, device)
        if not test_results:
            continue

        results.append(
            {
                "downsample_factor": factor,
                "test_accuracy": test_results["accuracy"],
                "test_auroc": test_results["auroc"],
            }
        )
        pd.DataFrame(results).to_csv(results_path, index=False)
        print(f"Saved results to {results_path}")

    print("\nChronos BBT downsample sweep complete.")
    if results:
        print(pd.DataFrame(results))


if __name__ == "__main__":
    main()
