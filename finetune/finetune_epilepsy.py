from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


from models.ffm_transformer import ModelConfig, TransformerFFM, NeuralFFMModel
from models.finetune_epilepsy_model import FinetuneModel
from data.UniMib_data.dataset.finetune_epilepsy_dataset import FinetuneEpilepsyDataset

torch.manual_seed(42)
np.float_ = np.float64

def train(model, train_loader, valid_loader, epochs, device, optimizer, scheduler):
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    
    train_acc_history, val_acc_history = [], []
    train_f1_history, val_f1_history = [], []

    patience = 15
    epochs_no_improve = 0
    best_val_acc = 0.0
    best_model_state = model.state_dict()
    stopped_epoch = epochs

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        train_probs_all, train_labels_all = [], []
        epoch_train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            with torch.no_grad():
                probs = torch.sigmoid(output).cpu().numpy()
                train_probs_all.append(probs)
                train_labels_all.append(labels.cpu().numpy())
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_labels = np.concatenate(train_labels_all)
        train_probs = np.concatenate(train_probs_all)
        train_preds = (train_probs > 0.5).astype(int)
        
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        train_acc_history.append(train_acc)
        train_f1_history.append(train_f1)
        
        model.eval()
        val_probs_all, val_labels_all = [], []
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                output = model(inputs)
                loss = loss_fn(output, labels)
                epoch_val_loss += loss.item()
                
                probs = torch.sigmoid(output).cpu().numpy()
                val_probs_all.append(probs)
                val_labels_all.append(labels.cpu().numpy())
        
        avg_val_loss = epoch_val_loss / len(valid_loader)
        val_labels = np.concatenate(val_labels_all)
        val_probs = np.concatenate(val_probs_all)
        val_preds = (val_probs > 0.5).astype(int)
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_acc_history.append(val_acc)
        val_f1_history.append(val_f1)
        
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}')
        print(f'Val Loss:   {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print(f"New best validation Accuracy: {best_val_acc:.4f}. Model state saved.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Stopping early. Best validation Accuracy was {best_val_acc:.4f}.")
            model.load_state_dict(best_model_state)
            stopped_epoch = epoch + 1
            break

        if scheduler:
            scheduler.step(avg_val_loss)
            
    model.load_state_dict(best_model_state)
    return train_acc_history, train_f1_history, val_acc_history, val_f1_history, stopped_epoch

def test(model, test_loader, device, feature_extraction_time):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            
            probs = torch.sigmoid(output).cpu().numpy()
            probs_all.append(probs)
            labels_all.append(labels.cpu().numpy())

    labels = np.concatenate(labels_all)
    probs = np.concatenate(probs_all)
    preds = (probs > 0.5).astype(int)
    
    test_accuracy = accuracy_score(labels, preds)
    test_f1 = f1_score(labels, preds, average='macro')

    print(f'Test Accuracy: {test_accuracy:.4f} | Test Macro F1: {test_f1:.4f} (Extraction Time: {feature_extraction_time})')
    return test_accuracy, test_f1


def _fgno_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path, root: Path | None = None) -> Path:
    root = root or _fgno_root()
    path = Path(path)
    return path if path.is_absolute() else root / path


def main(config_path: str | Path | None = None):
    """Run epilepsy fine-tuning. Default config preserves the original pipeline."""
    cli_conf = OmegaConf.from_cli()
    root = _fgno_root()
    default_cfg = root / "conf" / "finetune_epilepsy.yaml"
    yaml_path = Path(config_path) if config_path is not None else default_cfg
    if not yaml_path.is_absolute():
        yaml_path = root / yaml_path
    yaml_conf = OmegaConf.load(str(yaml_path))
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

    full_train_dataset = FinetuneEpilepsyDataset(DATA_DIR / cfg_env.data.dataset_file)
    
    train_data_len = len(full_train_dataset)
    train_subset_len = int(train_data_len * cfg_env.data.train_split)
    val_len = int(train_data_len * cfg_env.data.val_split)
    test_len = train_data_len - train_subset_len - val_len
    
    print(f"Dataset Split Summary:")
    print(f"   Total Samples: {train_data_len}")
    print(f"   Train ({cfg_env.data.train_split*100}%): {train_subset_len}")
    print(f"   Val ({cfg_env.data.val_split*100}%):   {val_len}")
    print(f"   Test (Remainder): {test_len}")

    train_dataset, val_dataset, test_dataset = random_split(
        full_train_dataset,
        [train_subset_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg_env.data.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg_env.data.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg_env.data.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("📏 Determining data dimensions for ModelConfig...")
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

    # 5. Main Execution Loop
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
                finetune_model.build_model(model_cfg, ffm_model, load_model['hidden_dim'], device, [layer_index])

                trainable_params = filter(lambda p: p.requires_grad, finetune_model.parameters())
                optimizer = torch.optim.AdamW(trainable_params, lr=cfg_env.train.lr, weight_decay=cfg_env.train.weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

                train(finetune_model, train_loader, val_loader, cfg_env.train.epochs, device, optimizer, scheduler)
                test_acc, test_f1 = test(finetune_model, test_loader, device, extraction_time)

                if noisy_input:
                    test_accs.append(test_acc)
                    test_f1s.append(test_f1)

                print(f"Finished run with seed {seed}. Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

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
                print(f"\nAggregated: Mean Acc: {mean_acc:.4f} ± {std_acc:.4f} | Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
                save_path = cfg_env.results.noisy_save_path
            else:
                results.append({
                    "layer_indices": str(layer_index),
                    "time": extraction_time,
                    "test_acc": test_acc,
                    "test_f1": test_f1
                })
                save_path = cfg_env.results.clean_save_path

            results_df = pd.DataFrame(results)
            results_df.to_csv(save_path, index=False)
            print(f"Saved intermediate results to {save_path}")

    print("\nAll Experiments Complete!")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()