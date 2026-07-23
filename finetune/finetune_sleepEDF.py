from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from models.ffm_transformer import ModelConfig, TransformerFFM, NeuralFFMModel
from data.UniMib_data.dataset.finetune_dataset import FinetuneUniMibDataset
from models.finetune_sleepEDF_model import FinetuneModel
torch.manual_seed(42)
np.float_ = np.float64


def train(model, train_loader, valid_loader, cfg, epochs, device, optimizer, scheduler):
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()  

    train_acc_history, val_acc_history = [], []
    train_f1_history, val_f1_history = [], []

    patience = 15
    epochs_no_improve = 0
    best_val_acc = 0.0
    best_model_state = model.state_dict()

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        epoch_train_loss = 0.0
        train_preds, train_labels_ls = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()  

            optimizer.zero_grad()
            outputs = model(inputs)  

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            train_preds.append(preds.cpu().numpy())
            train_labels_ls.append(labels.cpu().numpy())

        train_preds = np.concatenate(train_preds)
        train_labels_ls = np.concatenate(train_labels_ls)
        
        train_acc = accuracy_score(train_labels_ls, train_preds)
        train_f1 = f1_score(train_labels_ls, train_preds, average='macro')
        train_acc_history.append(train_acc)
        train_f1_history.append(train_f1)
        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_preds, val_labels_ls = [], []

        with torch.no_grad():
            for inputs, val_labels in valid_loader:
                inputs, val_labels = inputs.to(device), val_labels.to(device).long()
                outputs = model(inputs)

                loss = loss_fn(outputs, val_labels)
                epoch_val_loss += loss.item()

                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                val_preds.append(preds.cpu().numpy())
                val_labels_ls.append(val_labels.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_labels_ls = np.concatenate(val_labels_ls)
        
        val_acc = accuracy_score(val_labels_ls, val_preds)
        val_f1 = f1_score(val_labels_ls, val_preds, average='macro')
        val_acc_history.append(val_acc)
        val_f1_history.append(val_f1)
        avg_val_loss = epoch_val_loss / len(valid_loader)

        print(f"\nEpoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Stopping early. Best validation Accuracy was {best_val_acc:.4f}.")
                break

        if scheduler:
            scheduler.step(avg_val_loss)

    model.load_state_dict(best_model_state)
    return train_acc_history, train_f1_history, val_acc_history, val_f1_history


def test(model, test_loader, device, num_classes):
    model.eval()
    test_preds, test_labels_ls = [], []

    with torch.no_grad():
        for inputs, test_labels in test_loader:
            inputs, test_labels = inputs.to(device), test_labels.to(device).long()
            outputs = model(inputs)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            
            test_preds.append(preds.cpu().numpy())
            test_labels_ls.append(test_labels.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_labels_ls = np.concatenate(test_labels_ls)

    test_acc = accuracy_score(test_labels_ls, test_preds)
    test_f1 = f1_score(test_labels_ls, test_preds, average='macro')

    target_names = [f"Class {i}" for i in range(num_classes)] 
    print(classification_report(test_labels_ls, test_preds, target_names=target_names))
    print(f"✅ Test Accuracy: {test_acc:.4f} | Test Macro F1: {test_f1:.4f}")
    
    return test_acc, test_f1

def _fgno_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path, root: Path | None = None) -> Path:
    root = root or _fgno_root()
    path = Path(path)
    return path if path.is_absolute() else root / path


def main(config_path: str | Path | None = None):
    """Run Sleep-EDF fine-tuning. Default config preserves the original pipeline."""
    cli_conf = OmegaConf.from_cli()
    root = _fgno_root()
    default_cfg = root / "conf" / "finetune_sleepEDF.yaml"
    yaml_path = Path(config_path) if config_path is not None else default_cfg
    if not yaml_path.is_absolute():
        yaml_path = root / yaml_path
    try:
        yaml_conf = OmegaConf.load(str(yaml_path))
    except FileNotFoundError:
        print(f"YAML config not found: {yaml_path}")
        return

    cfg_env = OmegaConf.merge(yaml_conf, cli_conf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: {yaml_path}")

    DATA_DIR = _resolve_path(cfg_env.data.path, root)
    cfg_env.model.upstream_ckpt = str(_resolve_path(cfg_env.model.upstream_ckpt, root))
    cfg_env.results.clean_save_path = str(_resolve_path(cfg_env.results.clean_save_path, root))
    cfg_env.results.noisy_save_path = str(_resolve_path(cfg_env.results.noisy_save_path, root))
    Path(cfg_env.results.clean_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg_env.results.noisy_save_path).parent.mkdir(parents=True, exist_ok=True)

    print("\nCreating datasets and dataloaders...")
    full_dataset = FinetuneUniMibDataset(DATA_DIR / cfg_env.data.dataset_file, num_classes=cfg_env.model.num_classes)
    
    full_data_len = len(full_dataset)
    train_len = int(full_data_len * cfg_env.data.train_split)
    val_len = int(full_data_len * cfg_env.data.val_split)
    test_len = full_data_len - train_len - val_len 
    
    print(f"📊 Dataset Split Summary:")
    print(f"   Total Samples: {full_data_len}")
    print(f"   Train ({cfg_env.data.train_split*100}%): {train_len}")
    print(f"   Val ({cfg_env.data.val_split*100}%):   {val_len}")
    print(f"   Test (Remainder): {test_len}")
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42) 
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg_env.data.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg_env.data.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg_env.data.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    temp_data = torch.load(DATA_DIR / cfg_env.data.dim_check_file)['samples']
    seq_len = temp_data.shape[1]
    input_dim = temp_data.shape[2]

    model_cfg = ModelConfig(
        input_dim=input_dim,
        hidden_dim=768,
        num_heads=12,
        num_layers=6,
        feedforward_dim=3072,
        dropout=0.1,
        seq_len=seq_len,
        lr=cfg_env.train.lr
    )

    noisy_input = cfg_env.train.noisy_input
    if noisy_input:
        seeds = [s * 10 for s in range(42, 42 + cfg_env.train.num_runs)]
        print(f"Running in NOISY input mode. Performing {cfg_env.train.num_runs} runs with seeds: {seeds}")
    else:
        seeds = [123]
        print("Running in STANDARD (non-noisy) input mode.")

    results = []
    layer_indices = list(range(6))
    feature_extraction_time = np.linspace(0.0, 1, 10).tolist()
    for layer_index in layer_indices:
        for extraction_time in feature_extraction_time:
            test_accs, test_f1s = [], []
            
            for seed in seeds:
                print(f"\n{'='*20} STARTING RUN {'='*20}")
                print(f"Config: Layer=[{layer_index}], Time={extraction_time:.2f}, Seed={seed}")
                
                torch.manual_seed(seed)
                np.random.seed(seed)

                load_model = torch.load(cfg_env.model.upstream_ckpt, map_location=device)
                transformer_model = TransformerFFM(model_cfg, intermediate_rep=True)
                transformer_model.load_state_dict(load_model['model_state_dict'])
                ffm_model = NeuralFFMModel(transformer_model, model_cfg, device=device)

                finetune_model = FinetuneModel(feature_extraction_time=extraction_time, noisy_input=noisy_input)
                finetune_model.build_model(
                    model_cfg, ffm_model, load_model['hidden_dim'], device, cfg_env.model.num_classes, [layer_index]
                )

                trainable_params = filter(lambda p: p.requires_grad, finetune_model.parameters())
                optimizer = torch.optim.AdamW(trainable_params, lr=cfg_env.train.lr, weight_decay=cfg_env.train.weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

                train(finetune_model, train_loader, val_loader, model_cfg, cfg_env.train.epochs, device, optimizer, scheduler)
                
                print("\n🔬 Running final evaluation on test set...")
                test_acc, test_f1 = test(finetune_model, test_loader, device, cfg_env.model.num_classes)

                if noisy_input:
                    test_accs.append(test_acc)
                    test_f1s.append(test_f1)
                
                print(f"{'='*20} END OF RUN {'='*20}\n")

            if noisy_input:
                mean_acc, std_acc = np.mean(test_accs), np.std(test_accs)
                mean_f1, std_f1 = np.mean(test_f1s), np.std(test_f1s)
                results.append({
                    "layer_indices": str(layer_index),
                    "time": extraction_time,
                    "test_acc_mean": mean_acc,
                    "test_acc_std": std_acc,
                    "test_f1_mean": mean_f1,
                    "test_f1_std": std_f1
                })
                print(f"📈 Aggregated Result: Layer=[{layer_index}], Time={extraction_time:.2f} | Mean AUC: {mean_acc:.4f} ± {std_acc:.4f}")
                save_path = cfg_env.results.noisy_save_path
            else:
                results.append({
                    "layer_indices": str(layer_index),
                    "time": extraction_time,
                    "test_acc": test_acc,
                    "test_f1": test_f1
                })
                save_path = cfg_env.results.clean_save_path

            # Save incrementally after each configuration completes
            results_df = pd.DataFrame(results)
            results_df.to_csv(save_path, index=False)
            print(f"Saved intermediate results to {save_path}")

    print("\nAll Experiments Complete!")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()