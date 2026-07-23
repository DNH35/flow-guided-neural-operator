"""DREAMT skin-temperature regression finetuning, MAE variant.

Loads the masked-autoencoder checkpoint from pretrain/pretrain_mae_dreamt.py
(checkpoints/mae_dreamt_BVP_ACC_upsample_epoch_<N>.pt via
conf/dreamt_skin_mae_finetune.yaml) and predicts skin temperature from
BVP+ACC windows. FFM-side counterpart: finetune_dreamt_BVP_HR.py.

Note on the regression target: data_preprocess/dreamt/preprocess_dreamt_subject_splits.py
pairs each window with a 20-sample (4Hz) skin-temperature *sequence*, but
FinetuneRegressionModel (models/finetune_regression_dreamt_mae_model.py)
predicts a single scalar per window. We reduce the sequence to its mean here
-- "the average skin temperature over this window" -- rather than changing
the model to predict a sequence.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch_optimizer as torch_optim

from models.model_config import ModelConfig
from models.dreamt_mae import MaskedAutoencoderTransformer
from models.finetune_regression_dreamt_mae_model import FinetuneRegressionModel
from models.gaussian_normalizer import GaussianNormalizer

torch.manual_seed(42)

def extract_dataset(obj):
    if isinstance(obj, torch.utils.data.DataLoader):
        return obj.dataset
    return obj

def scalar_skin_temp_target(labels: torch.Tensor) -> torch.Tensor:
    """Collapse a (batch, 20) skin-temperature sequence label to a (batch,)
    scalar (its mean), matching FinetuneRegressionModel's scalar output.
    """
    return labels.mean(dim=1) if labels.dim() > 1 else labels

def train_regression(model, train_loader, valid_loader, epochs, device, optimizer, scheduler, normalizer):
    model = model.to(device)
    loss_fn = nn.L1Loss()
    
    history = {'train_mae': [], 'val_mae': [], 'val_mse': [], 'val_rmse': []}
    best_val_mae = float('inf')
    patience = 15
    epochs_no_improve = 0
    best_model_state = model.state_dict()

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        train_predicts, train_labels = [], []
        
        for batch in train_loader:
            inputs = batch['input'].to(device)
            labels = scalar_skin_temp_target(batch['labels'].to(device))
            optimizer.zero_grad()
            
            normalized_output = model(inputs)
            
            normalized_labels = normalizer.encode(labels)  
            
            loss = loss_fn(normalized_output, normalized_labels)
            loss.backward()
            optimizer.step()
            
            decoded_output = normalizer.decode(normalized_output)
            train_predicts.extend(decoded_output.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        if len(train_predicts) > 0:
            print(f"Sample Predictions: {train_predicts[:5]}")
            print(f"Sample Labels: {train_labels[:5]}")
        
        history['train_mae'].append(mean_absolute_error(train_labels, train_predicts))
        
        model.eval()
        val_predicts, val_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                inputs = batch['input'].to(device)
                labels = scalar_skin_temp_target(batch['labels'].to(device))

                normalized_output = model(inputs)
                decoded_output = normalizer.decode(normalized_output)
                val_predicts.extend(decoded_output.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_mae = mean_absolute_error(val_labels, val_predicts)
        val_mse = mean_squared_error(val_labels, val_predicts)
        val_rmse = np.sqrt(val_mse)
        
        history['val_mae'].append(val_mae)
        history['val_mse'].append(val_mse)
        history['val_rmse'].append(val_rmse)

        print(f'\nEpoch {epoch+1}/{epochs} -> Val MAE: {val_mae:.3f} | Val RMSE: {val_rmse:.3f} | Val MSE: {val_mse:.3f}')

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print(f"New best validation MAE: {best_val_mae:.4f}. Model saved.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Stopping early. Best validation MAE was {best_val_mae:.4f}.")
            model.load_state_dict(best_model_state)
            break

        if scheduler:
            scheduler.step(val_mae)

    model.load_state_dict(best_model_state)
    return history

def test_regression(model, test_loader, device, normalizer):
    """Test function that decodes predictions before calculating metrics."""
    model.to(device)
    model.eval()
    predicts, labels_ls = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            labels = scalar_skin_temp_target(batch['labels'].to(device))
            normalized_output = model(inputs)
            decoded_output = normalizer.decode(normalized_output)
            predicts.extend(decoded_output.detach().cpu().numpy())
            labels_ls.extend(labels.detach().cpu().numpy())

    test_mae = mean_absolute_error(labels_ls, predicts)
    test_mse = mean_squared_error(labels_ls, predicts)
    test_rmse = np.sqrt(test_mse)
    
    print(f"Mean Absolute Error (MAE):  {test_mae:.4f} °C")
    print(f"Mean Squared Error (MSE):   {test_mse:.4f} °C²")
    print(f"Root Mean Squared (RMSE): {test_rmse:.4f} °C")
        
    return {'mae': test_mae, 'mse': test_mse, 'rmse': test_rmse}

if __name__ == "__main__":
    
    cfg_env = OmegaConf.load(str(Path(__file__).resolve().parents[2] / "conf/dreamt_skin_mae_finetune.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    BATCH_SIZE = cfg_env.data.batch_size
    SAVE_DIR = Path(cfg_env.data.path)

    train_loader = torch.load(SAVE_DIR / cfg_env.data.train_path, weights_only=False)
    val_loader = torch.load(SAVE_DIR / cfg_env.data.val_path, weights_only=False)
    test_loader = torch.load(SAVE_DIR / cfg_env.data.test_path, weights_only=False)

    if cfg_env.data.low_data_mode:
        d_train = extract_dataset(train_loader)
        d_val = extract_dataset(val_loader)
        d_test = extract_dataset(test_loader)

        full_dataset = ConcatDataset([d_train, d_val, d_test])
        full_data_len = len(full_dataset)

        train_len = int(full_data_len * 0.05)
        val_len = int(full_data_len * 0.10)
        test_len = full_data_len - train_len - val_len

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    all_train_labels = []
    for batch in tqdm(train_loader, desc="Extracting labels"):
        all_train_labels.append(scalar_skin_temp_target(batch['labels']))
    all_train_labels_tensor = torch.cat(all_train_labels)
    
    normalizer = GaussianNormalizer(all_train_labels_tensor)
    normalizer.to(device)
    print(f"Mean: {normalizer.mean.item():.4f}, Std: {normalizer.std.item():.4f}")

    cfg = ModelConfig(
        input_dim=132,
        hidden_dim=768,
        num_heads=12,
        num_layers=6,
        feedforward_dim=3072,
        dropout=0.1,
        seq_len=21,
    )

    results = []
    layer_indices_list = list(range(6))
    
    for layer in layer_indices_list:
        mae_backbone = MaskedAutoencoderTransformer(cfg).to(device)
        mae_backbone.load_state_dict(torch.load(cfg_env.model.upstream_ckpt, map_location=device))

        finetune_model = FinetuneRegressionModel()
        finetune_model.build_model(
            cfg=cfg,
            backbone_model=mae_backbone,
            hidden_dim=cfg.hidden_dim,
            seq_len=cfg.seq_len,
            device=device,
            layer_indices=[layer]
        )
        
        trainable_params = filter(lambda p: p.requires_grad, finetune_model.parameters())
        optimizer = torch_optim.Lamb(trainable_params, lr=0.0001, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

        training_history = train_regression(
            finetune_model, train_loader, val_loader, epochs, device, optimizer, scheduler, normalizer
        )        
        
        test_metrics = test_regression(
            finetune_model, test_loader, device, normalizer
        )

        results.append({
            'layer': layer,
            'test_mae': test_metrics['mae'],
            'test_mse': test_metrics['mse'],
            'test_rmse': test_metrics['rmse'],
        })

        # Save Iterative Results
        results_df = pd.DataFrame(results)
        results_df.to_csv(cfg_env.results.save_path, index=False)
        print(f"Results saved to '{cfg_env.results.save_path}'.")
        print(f"Finished training and testing for Layer {layer}")