import time
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from sklearn.metrics import roc_auc_score

from models.ffm_transformer import ModelConfig, TransformerFFM, NeuralFFMModel
from models.finetune_ffm_model import FinetuneFFMModel

torch.manual_seed(42)

def extract_dataset(obj):
    if isinstance(obj, torch.utils.data.DataLoader):
        return obj.dataset
    return obj

def train(model, train_loader, valid_loader, epochs, device, optimizer, scheduler):
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        
        for batch in train_loader:
            inputs = batch['input'].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_predicts, val_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                inputs = batch['input'].to(device)
                labels = batch["labels"].to(device)
                output = model(inputs)
                probs = torch.sigmoid(output).cpu().numpy()
                val_predicts.append(probs)
                val_labels.append(labels.cpu().numpy())

        val_labels = np.concatenate(val_labels)
        val_predicts = np.concatenate(val_predicts)
        val_roc_auc = roc_auc_score(val_labels, val_predicts)

        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}/{epochs} | Val AUC: {val_roc_auc:.4f} | Epoch Time: {epoch_time:.2f} sec')

        if scheduler:
            scheduler.step(val_roc_auc)

    total_time = time.time() - total_start_time
    print(f'\n⏱️ Total fine-tuning time for {epochs} epochs: {total_time/60:.2f} min')

def test(model, test_loader, device):
    model.eval()
    model.to(device)
    predicts, labels_ls = [], []
    total_start_time = time.time()
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            labels = batch["labels"].to(device)
            output = model(inputs)
            predict = torch.sigmoid(output).cpu().numpy()
            predicts.append(predict)
            labels_ls.append(labels.cpu().numpy())
            
    labels_ls = np.concatenate(labels_ls)
    predicts = np.concatenate(predicts)
    roc_auc = roc_auc_score(labels_ls, predicts)
    
    print(f'Test ROC AUC: {roc_auc:.4f} (Inference at time t={model.feature_extraction_time})')
    print(f'⏱️ Total testing time: {(time.time() - total_start_time):.2f} sec')
    return roc_auc


if __name__ == '__main__':
    
    cfg_env = OmegaConf.load(str(Path(__file__).resolve().parents[2] / 'conf/dreamt_sleep_finetune.yaml'))

    BATCH_SIZE = cfg_env.data.batch_size
    SAVE_DIR = Path(cfg_env.data.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🔄 Loading existing data files...")
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

        print(f"📊 Dataset Split Summary:")
        print(f"   Total Samples: {full_data_len}")
        print(f"   Train (5%):    {train_len}")
        print(f"   Val (10%):     {val_len}")
        print(f"   Test (85%):    {test_len}")

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    sample_batch = train_loader.dataset[0]['input']
    input_dim = sample_batch.shape[-1]
    seq_len = sample_batch.shape[-2]

    EPOCHS = 100

    model_cfg = ModelConfig(
        input_dim=input_dim,
        hidden_dim=768,
        num_heads=12,
        num_layers=6,
        feedforward_dim=3072,
        dropout=0.1,
        seq_len=seq_len,
        lr=0.0001
    )

    results = []
    layer_indices_to_use = list(range(6))
    feature_extraction_time_set = np.linspace(0.0, 1, 10).tolist()
    inference_time = 1.0

    for layer_idx in layer_indices_to_use:
        for t_train in feature_extraction_time_set:
            print("\n" + "="*50)
            print(f"STARTING RUN: Training at time t={t_train:.2f} on Layer {layer_idx}")
            print("="*50)

            load_model = torch.load(cfg_env.model.upstream_ckpt)
            transformer_model = TransformerFFM(model_cfg, intermediate_rep=True)
            hidden_dim = load_model['hidden_dim']
            transformer_model.load_state_dict(load_model['model_state_dict'])
            ffm_model = NeuralFFMModel(transformer_model, model_cfg, device=device)

            finetune_model = FinetuneFFMModel(feature_extraction_time=t_train)
            finetune_model.build_model(model_cfg, ffm_model, hidden_dim, device, [layer_idx])
            
            trainable_params = filter(lambda p: p.requires_grad, finetune_model.parameters())
            optimizer = torch.optim.Adam(trainable_params, lr=model_cfg.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

            print(f"Training model with features from t={t_train:.2f}...")
            train(finetune_model, train_loader, val_loader, EPOCHS, device, optimizer, scheduler)

            print(f"\nSwitching model's feature extraction time to t={inference_time} for testing.")
            finetune_model.feature_extraction_time = inference_time

            test_roc_auc = test(finetune_model, test_loader, device)
            results.append({
                "layer_idx": layer_idx,
                "training_time": t_train,
                "test_roc_auc_at_t1": test_roc_auc
            })
            
            # Clean up memory
            del finetune_model, ffm_model, load_model, optimizer, scheduler
            torch.cuda.empty_cache()

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg_env.results.save_path, index=False)
    print("\nExperiment Complete! Results saved to:", cfg_env.results.save_path)
    print(results_df)