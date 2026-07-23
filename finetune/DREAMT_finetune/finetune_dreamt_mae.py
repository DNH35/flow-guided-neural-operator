import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from models.model_config import ModelConfig
from models.dreamt_mae import MaskedAutoencoderTransformer
from models.finetune_classification_dreamt_mae_model import FinetuneModel

torch.manual_seed(42)

def extract_dataset(obj):
    if isinstance(obj, torch.utils.data.DataLoader):
        return obj.dataset
    return obj

def train(model, train_loader, valid_loader, cfg, epochs, device, optimizer, scheduler):
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    
    train_auc_history, val_auc_history = [], []
    total_start_time = time.time()

    for epoch in tqdm(range(epochs)):
        epoch_start_time = time.time()

        model.train()
        train_predicts, train_labels = [], []
        epoch_train_loss = 0.0
        
        for batch in train_loader:
            inputs = batch['input'].to(device)
            labels = torch.FloatTensor(batch["labels"]).to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            if output.dim() > 1 and output.size(-1) == 1:
                output = output.squeeze(-1)
            
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            with torch.no_grad():
                probs = torch.sigmoid(output).cpu().numpy()
                train_predicts.append(probs)
                train_labels.append(labels.cpu().numpy())
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_labels = np.concatenate(train_labels)
        train_predicts = np.concatenate(train_predicts)
        train_roc_auc = roc_auc_score(train_labels, train_predicts)
        train_auc_history.append(train_roc_auc)
        
        model.eval()
        val_predicts, val_labels = [], []
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch in valid_loader:
                inputs = batch['input'].to(device)
                labels = torch.FloatTensor(batch["labels"]).to(device)
                
                output = model(inputs)
                if output.dim() > 1 and output.size(-1) == 1:
                    output = output.squeeze(-1)
                
                loss = loss_fn(output, labels)
                epoch_val_loss += loss.item()
                
                probs = torch.sigmoid(output).cpu().numpy()
                val_predicts.append(probs)
                val_labels.append(labels.cpu().numpy())
        
        avg_val_loss = epoch_val_loss / len(valid_loader)
        val_labels = np.concatenate(val_labels)
        val_predicts = np.concatenate(val_predicts)
        val_roc_auc = roc_auc_score(val_labels, val_predicts)
        val_auc_history.append(val_roc_auc)

        epoch_time = time.time() - epoch_start_time

        print(f'\nEpoch {epoch+1}/{epochs} | '
              f'Train Loss: {avg_train_loss:.4f} | Train AUC: {train_roc_auc:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | Val AUC: {val_roc_auc:.4f} | '
              f'Epoch Time: {epoch_time:.2f} sec')

        if scheduler:
            scheduler.step(avg_val_loss)

    total_time = time.time() - total_start_time
    print(f"\n Total fine-tuning time for {epochs} epochs: {total_time/60:.2f} minutes")

    return train_auc_history, val_auc_history


def test(model, test_loader, device):
    model.eval()
    total_start_time = time.time()
    predicts, labels_ls = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            labels = torch.FloatTensor(batch["labels"]).to(device)
            
            output = model.forward(inputs)
            predict = torch.sigmoid(output).squeeze().cpu().numpy()
            
            predicts.append(predict)
            labels_ls.append(labels.cpu().numpy())
            
    total_end_time = time.time()
    labels_ls = np.concatenate(labels_ls)
    predicts = np.concatenate(predicts)
    roc_auc = roc_auc_score(labels_ls, predicts)

    print(f'Test ROC AUC: {roc_auc:.4f}')
    print(f'⏱️ Total testing time: {(total_end_time - total_start_time):.2f} sec')
    return roc_auc

if __name__ == "__main__":
    
    cfg_env = OmegaConf.load(str(Path(__file__).resolve().parents[2] / "conf/dreamt_sleep_mae_finetune.yaml"))
    BATCH_SIZE = cfg_env.data.batch_size
    EPOCHS = 50
    SAVE_DIR = Path(cfg_env.data.path)
    MAE_MODEL_PATH = cfg_env.model.upstream_ckpt
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig(
        input_dim=132,
        hidden_dim=768,
        num_heads=12,
        num_layers=6,
        feedforward_dim=3072,
        dropout=0.1,
        seq_len=21,
    )

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
        val_len = int(full_data_len * 0.05)
        test_len = full_data_len - train_len - val_len

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    results = []
    layer_indices_list = list(range(6)) 
    
    for layer in layer_indices_list:
        print(f"\n{'='*20} FINE-TUNING ON LAYER {layer} {'='*20}")        
        mae_backbone = MaskedAutoencoderTransformer(cfg).to(device)
        mae_backbone.load_state_dict(torch.load(MAE_MODEL_PATH, map_location=device))
        print(f"   - Weights loaded from: {MAE_MODEL_PATH}")

        finetune_model = FinetuneModel()
        finetune_model.build_model(
            cfg=cfg,
            backbone_model=mae_backbone,
            hidden_dim=cfg.hidden_dim,
            device=device,
            layer_indices=[layer]
        )
        
        trainable_params = filter(lambda p: p.requires_grad, finetune_model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

        train_auc_history, val_auc_history = train(finetune_model, train_loader, val_loader, cfg, EPOCHS, device, optimizer, scheduler)
        
        test_auc = test(finetune_model, test_loader, device)

        results.append({
            "layer_indices": str(layer),
            "test_acc": test_auc
        })

        results_df = pd.DataFrame(results)
        results_df.to_csv(cfg_env.results.save_path, index=False)
        print(results_df)