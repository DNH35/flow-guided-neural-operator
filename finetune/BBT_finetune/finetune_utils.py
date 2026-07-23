"""Shared Brain Treebank (BBT) fine-tuning helpers for FGNO release scripts."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, random_split
from tqdm import tqdm

import tasks
from models.ffm_transformer import ModelConfig, NeuralFFMModel, TransformerFFM
from models.finetune_ffm_model import FinetuneFFMModel, ModelEMA

FGNO_ROOT = Path(__file__).resolve().parents[2]


def ensure_fgno_on_path() -> Path:
    """Make FGNO importable when scripts are launched from any cwd."""
    root = str(FGNO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return FGNO_ROOT


def resolve_path(path: str | Path, base: Optional[Path] = None) -> Path:
    """Resolve a config path relative to FGNO root unless already absolute."""
    path = Path(os.path.expanduser(str(path)))
    if path.is_absolute():
        return path
    return (base or FGNO_ROOT) / path


def load_cfg(config_name: str) -> DictConfig:
    """Load a YAML config from FGNO/conf."""
    ensure_fgno_on_path()
    cfg_path = FGNO_ROOT / "conf" / config_name
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    # Normalize absolute-ish paths relative to FGNO root when possible.
    if "raw_brain_data_dir" in cfg.data:
        cfg.data.raw_brain_data_dir = str(resolve_path(cfg.data.raw_brain_data_dir))
    if "upstream_ckpt" in cfg.model:
        cfg.model.upstream_ckpt = str(resolve_path(cfg.model.upstream_ckpt))
    return cfg


def maybe_init_wandb(project: str, run_name: str, enabled: Optional[bool] = None):
    """Initialize wandb if available and not disabled via FGNO_WANDB=0."""
    if enabled is None:
        enabled = os.environ.get("FGNO_WANDB", "1") != "0"
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        logging.warning("wandb not installed; continuing without logging.")
        return None

    init_kwargs = {
        "project": os.environ.get("FGNO_WANDB_PROJECT", project),
        "name": run_name,
    }
    entity = os.environ.get("FGNO_WANDB_ENTITY") or os.environ.get("WANDB_ENTITY")
    if entity:
        init_kwargs["entity"] = entity
    return wandb.init(**init_kwargs)


def get_electrode_dataloader(cfg: DictConfig, electrode: str):
    """Build train/valid/test iterators for a single electrode."""
    cfg.data.electrodes = [electrode]
    logging.info("Running for electrode: %s", electrode)

    task = tasks.setup_task(cfg.exp.task)
    task.load_datasets(cfg.data, cfg.data.preprocessor)

    def get_batch_iterator(dataset, batch_size, **kwargs):
        return task.get_batch_iterator(dataset, batch_size, **kwargs)

    train_loader = get_batch_iterator(
        task.train_set,
        cfg.exp.runner.train_batch_size,
        shuffle=True,
        num_workers=cfg.exp.runner.num_workers,
    )
    valid_loader = get_batch_iterator(
        task.valid_set,
        cfg.exp.runner.valid_batch_size,
        shuffle=cfg.exp.runner.shuffle,
        num_workers=cfg.exp.runner.num_workers,
    )
    test_loader = get_batch_iterator(
        task.test_set,
        cfg.exp.runner.valid_batch_size,
        shuffle=cfg.exp.runner.shuffle,
        num_workers=cfg.exp.runner.num_workers,
    )
    logging.info(
        "Samples - train: %d, valid: %d, test: %d",
        len(task.train_set),
        len(task.valid_set),
        len(task.test_set),
    )
    return train_loader, valid_loader, test_loader


def maybe_apply_low_data_mode(cfg: DictConfig, train_loader, valid_loader, test_loader):
    """Optionally collapse splits into a low-data 5/10/85 regime."""
    if not getattr(cfg.data, "low_data_mode", False):
        return train_loader, valid_loader, test_loader

    task_collate_fn = train_loader.collate_fn

    def extract_dataset(obj):
        if isinstance(obj, torch.utils.data.DataLoader):
            return obj.dataset
        return obj

    full_dataset = ConcatDataset(
        [
            extract_dataset(train_loader),
            extract_dataset(valid_loader),
            extract_dataset(test_loader),
        ]
    )
    full_data_len = len(full_dataset)
    train_len = int(full_data_len * 0.05)
    val_len = int(full_data_len * 0.10)
    test_len = full_data_len - train_len - val_len

    print(
        f"Low-data split | total={full_data_len} "
        f"train={train_len} val={val_len} test={test_len}"
    )
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.exp.runner.train_batch_size,
        shuffle=True,
        num_workers=cfg.exp.runner.num_workers,
        pin_memory=True,
        collate_fn=task_collate_fn,
    )
    valid_loader = DataLoader(
        val_dataset,
        batch_size=cfg.exp.runner.valid_batch_size,
        shuffle=False,
        num_workers=cfg.exp.runner.num_workers,
        pin_memory=True,
        collate_fn=task_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.exp.runner.valid_batch_size,
        shuffle=False,
        num_workers=cfg.exp.runner.num_workers,
        pin_memory=True,
        collate_fn=task_collate_fn,
    )
    return train_loader, valid_loader, test_loader


def train(
    model,
    train_loader,
    valid_loader,
    epochs: int,
    device,
    optimizer,
    scheduler=None,
    layer_indices=None,
    time_val: float = 0.5,
    ema_decay: float = 0.999,
    patience: int = 20,
    model_save_path: str = "best_model.pth",
    wandb_project: str = "fgno",
    run_prefix: str = "finetune",
):
    """Train a linear probe with EMA validation and early stopping."""
    model = model.to(device)
    ema = ModelEMA(model, decay=ema_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    run = maybe_init_wandb(
        project=wandb_project,
        run_name=f"{run_prefix}_layer_{layer_indices}_time_{time_val:.2f}",
    )
    if run is not None:
        import wandb

        wandb.watch(model, loss_fn, log="all", log_freq=100)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    save_parent = Path(model_save_path).parent
    if str(save_parent) not in ("", "."):
        save_parent.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        epoch_train_loss = 0.0
        train_predicts, train_labels = [], []

        for batch in tqdm(train_loader, desc="Training Batches", leave=False):
            inputs = batch["input"].to(device)
            labels = torch.FloatTensor(batch["labels"]).to(device)

            optimizer.zero_grad()
            output = model(inputs).squeeze(-1)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            ema.update()

            epoch_train_loss += loss.item()
            with torch.no_grad():
                train_predicts.append(torch.sigmoid(output).cpu().numpy())
                train_labels.append(labels.cpu().numpy())

        model.eval()
        ema.apply_shadow()
        epoch_val_loss = 0.0
        val_predicts, val_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                inputs = batch["input"].to(device)
                labels = torch.FloatTensor(batch["labels"]).to(device)
                output = model(inputs).squeeze(-1)
                epoch_val_loss += loss_fn(output, labels).item()
                val_predicts.append(torch.sigmoid(output).cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        avg_train_loss = epoch_train_loss / max(len(train_loader), 1)
        avg_val_loss = epoch_val_loss / max(len(valid_loader), 1)
        train_roc_auc = roc_auc_score(np.concatenate(train_labels), np.concatenate(train_predicts))
        val_roc_auc = roc_auc_score(np.concatenate(val_labels), np.concatenate(val_predicts))
        print(
            f"\nEpoch {epoch + 1}/{epochs} | Val AUC (EMA): {val_roc_auc:.4f} | "
            f"Val Loss (EMA): {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            print(f"New best EMA validation loss: {best_val_loss:.4f}. Saving to {model_save_path}")
            if run is not None:
                import wandb

                wandb.run.summary["best_val_auc"] = val_roc_auc
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1

        ema.restore()

        if run is not None:
            import wandb

            wandb.log(
                {
                    "loss/train_epoch": avg_train_loss,
                    "loss/val_epoch_ema": avg_val_loss,
                    "auc/train": train_roc_auc,
                    "auc/val_ema": val_roc_auc,
                    "epoch": epoch,
                },
                step=epoch,
            )

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement.")
            break

    if run is not None:
        import wandb

        wandb.finish()
    print(f"Training finished. Best model: {model_save_path} (val loss={best_val_loss:.4f})")
    return model_save_path


def test(model, test_loader, feature_extraction_time, device) -> float:
    """Evaluate ROC-AUC on the test split."""
    model.eval()
    predicts, labels_ls = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = batch["input"].to(device)
            output = model(inputs).squeeze(-1)
            labels = torch.FloatTensor(batch["labels"]).to(output.device)
            predict = torch.sigmoid(output).squeeze().detach().cpu().numpy()
            predicts.append(predict)
            labels_ls.append(labels.detach().cpu().numpy())

    labels_ls = np.array([x for y in labels_ls for x in y])
    predicts = [np.array([p]) if np.ndim(p) == 0 else p for p in predicts]
    predicts = np.concatenate(predicts)
    roc_auc = roc_auc_score(labels_ls, predicts)
    print(f"Test ROC AUC: {roc_auc:.4f} at flow time {feature_extraction_time}")
    return roc_auc


def build_finetune_model(
    cfg: DictConfig,
    model_cfg: ModelConfig,
    device,
    layer_indices: Sequence[int],
    time_val: float,
    hidden_dim: int,
    dropout_p: float = 0.2,
):
    """Instantiate upstream FFM + fine-tune head for one experiment cell."""
    saved_model_path = cfg.model.upstream_ckpt
    if not Path(saved_model_path).exists():
        raise FileNotFoundError(
            f"Upstream checkpoint not found: {saved_model_path}. "
            "Update conf/*.yaml model.upstream_ckpt."
        )

    transformer_model = TransformerFFM(model_cfg, intermediate_rep=True)
    transformer_model.load_state_dict(torch.load(saved_model_path, map_location="cpu")["model_state_dict"])
    ffm_model = NeuralFFMModel(transformer_model, cfg, device=device)

    finetune_model = FinetuneFFMModel(feature_extraction_time=time_val)
    finetune_model.build_model(
        cfg,
        ffm_model,
        hidden_dim,
        device,
        layer_indices=list(layer_indices),
        dropout_p=dropout_p,
    )
    return finetune_model


def run_bbt_finetune_sweep(
    config_name: str,
    task_name: str,
    results_default: str,
    layer_sets: Optional[List[List[int]]] = None,
    epochs: Optional[int] = None,
    lr: float = 5e-4,
    early_stopping_patience: int = 20,
    checkpoint_dir: str = "outputs/finetune_ckpts",
):
    """Common electrode × layer × flow-time fine-tuning sweep used by BBT scripts."""
    ensure_fgno_on_path()
    logging.basicConfig(level=logging.INFO)

    cfg = load_cfg(config_name)
    hidden_dim = int(getattr(cfg.model, "hidden_dim", 768))
    seq_len = int(getattr(cfg.model, "seq_len", 196))
    num_heads = int(getattr(cfg.model, "nhead", 12))
    num_layers = int(getattr(cfg.model, "upstream_num_layers", 6))
    feedforward_dim = int(getattr(cfg.model, "layer_dim_feedforward", 3072))
    dropout = float(getattr(cfg.model, "dropout", 0.2))
    if epochs is None:
        epochs = int(cfg.train.epochs) if hasattr(cfg, "train") and "epochs" in cfg.train else 100

    if layer_sets is None:
        layer_sets = [[i] for i in range(num_layers)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    electrode_names = list(cfg.data.electrodes)
    model_input_dim = cfg.data.preprocessor.target_freq_bins
    model_cfg = ModelConfig(
        input_dim=model_input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        seq_len=seq_len,
    )

    results_key = "results" if "results" in cfg else "result"
    results_save_path = cfg[results_key].save_path if results_key in cfg else results_default
    results_save_path = str(resolve_path(results_save_path))
    Path(results_save_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt_root = resolve_path(checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    feature_extraction_time_set = np.linspace(0.0, 1.0, 10).tolist()
    results = []

    for electrode in electrode_names:
        train_loader, valid_loader, test_loader = get_electrode_dataloader(cfg, electrode)
        train_loader, valid_loader, test_loader = maybe_apply_low_data_mode(
            cfg, train_loader, valid_loader, test_loader
        )

        for layer_indices in layer_sets:
            for time_val in feature_extraction_time_set:
                model_save_path = str(
                    ckpt_root
                    / f"{task_name}_e_{electrode}_layers_{layer_indices}_time_{time_val:.2f}.pth"
                )
                print(f"--- Starting experiment: {model_save_path} ---")

                finetune_model = build_finetune_model(
                    cfg, model_cfg, device, layer_indices, time_val, hidden_dim
                )
                optimizer = torch.optim.AdamW(finetune_model.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", patience=5
                )

                train(
                    finetune_model,
                    train_loader,
                    valid_loader,
                    epochs,
                    device,
                    optimizer,
                    scheduler,
                    layer_indices=layer_indices,
                    time_val=time_val,
                    ema_decay=0.999,
                    patience=early_stopping_patience,
                    model_save_path=model_save_path,
                    run_prefix=f"{task_name}_{electrode}",
                )

                finetune_model.load_state_dict(
                    torch.load(model_save_path, map_location="cpu")
                )
                finetune_model.to(device)
                test_acc = test(finetune_model, test_loader, time_val, device)

                results.append(
                    {
                        "electrode": electrode,
                        "layer_indices": str(layer_indices),
                        "time": time_val,
                        "test_acc": test_acc,
                        "model_path": model_save_path,
                    }
                )
                results_df = pd.DataFrame(results)
                results_df.to_csv(results_save_path, index=False)
                print("\nExperiment Results:")
                print(results_df)

    return results
