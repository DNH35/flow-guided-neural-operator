from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from chronos import ChronosPipeline
import warnings
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from omegaconf import OmegaConf
from imblearn.over_sampling import SMOTE



warnings.filterwarnings("ignore")

SAMPLING_RATE = 64
EPOCH_SECONDS = 5
SAMPLES_PER_EPOCH = SAMPLING_RATE * EPOCH_SECONDS

def preprocess_BVP(bvp_64hz: np.ndarray) -> np.ndarray:
    lowcut, highcut, order = 0.5, 20.0, 5
    sos = signal.cheby2(N=order, rs=30, Wn=[lowcut, highcut], btype='bandpass', fs=SAMPLING_RATE, output='sos')
    bvp_filtered = signal.sosfilt(sos, bvp_64hz)
    bvp_normalized = (bvp_filtered - np.mean(bvp_filtered)) / (np.std(bvp_filtered) + 1e-8)
    return bvp_normalized

def preprocess_ACC(acc_64hz_raw: np.ndarray) -> np.ndarray:
    acc_32hz = acc_64hz_raw[::2]
    sos = signal.butter(N=3, Wn=[3, 10], btype='bp', fs=32, output='sos')
    acc_32hz_filtered = signal.sosfilt(sos, acc_32hz)
    original_len, target_len = len(acc_32hz_filtered), len(acc_64hz_raw)
    original_time = np.linspace(0, original_len - 1, original_len)
    target_time = np.linspace(0, original_len - 1, target_len)
    interpolator = interp1d(original_time, acc_32hz_filtered, kind='linear', fill_value="extrapolate")
    acc_64hz_interpolated = interpolator(target_time)
    acc_normalized = (acc_64hz_interpolated - np.mean(acc_64hz_interpolated)) / (np.std(acc_64hz_interpolated) + 1e-8)
    return acc_normalized

def is_epoch_artifact(epoch_df: pd.DataFrame) -> bool:
    bvp = epoch_df.BVP.to_numpy()
    b, a = butter(N=2, Wn=[0.5 / (0.5 * SAMPLING_RATE), 15 / (0.5 * SAMPLING_RATE)], btype="band")
    filtered_signal = filtfilt(b, a, bvp)
    signal_power = np.mean(filtered_signal**2)
    noise_power = np.mean((bvp - filtered_signal)**2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    acc_x = (epoch_df.ACC_X.to_numpy() / 64)[::2]
    acc_y = (epoch_df.ACC_Y.to_numpy() / 64)[::2]
    acc_z = (epoch_df.ACC_Z.to_numpy() / 64)[::2]
    acc_std = np.std(np.sqrt(acc_x**2 + acc_y**2 + acc_z**2))
    if acc_std >= (0.4125 / 2) or snr_db < 10 or np.max(bvp) > 500 or np.min(bvp) < -500:
        return True
    return False

class ChronosEmbeddingDataset(Dataset):
    def __init__(self, raw_epochs, labels, chronos_pipeline):
        self.raw_epochs = raw_epochs
        self.labels = labels
        self.pipeline = chronos_pipeline
        self.num_channels = raw_epochs.shape[2]

    def __len__(self):
        return len(self.raw_epochs)

    def __getitem__(self, idx):
        epoch_data = self.raw_epochs[idx]
        label = self.labels[idx]

        epoch_transposed = np.transpose(epoch_data, (1, 0)) 
        context = torch.tensor(epoch_transposed, dtype=torch.bfloat16)

        with torch.no_grad():
            embeddings_per_channel, _ = self.pipeline.embed(context)
        
        final_epoch_embedding = embeddings_per_channel.mean(axis=0)
        final_epoch_embedding_flat = final_epoch_embedding.reshape(-1)
        detached_embedding = final_epoch_embedding_flat.detach()

        return detached_embedding.to(torch.float32), torch.tensor(label, dtype=torch.float32)


class ClassifierHead(nn.Module):
    def __init__(self, input_dim):
        super(ClassifierHead, self).__init__()
        head_input_dim = input_dim
        self.linear_out = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Linear(head_input_dim // 2, head_input_dim // 4),
            nn.ReLU(),
            nn.Linear(head_input_dim // 4, 1)
        )
    def forward(self, x):
        return self.linear_out(x)


def test_model(model, test_loader, device):
    model.eval()
    test_labels = []
    test_predicts_proba = []
    test_predicts_binary = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.unsqueeze(1).to(device)
            outputs = model(inputs)
            
            probas = torch.sigmoid(outputs)
            preds = torch.round(probas)

            test_labels.extend(labels.cpu().numpy())
            test_predicts_proba.extend(probas.cpu().numpy())
            test_predicts_binary.extend(preds.cpu().numpy())
            
    if len(np.unique(test_labels)) > 1:
        accuracy = accuracy_score(test_labels, test_predicts_binary)
        auroc = roc_auc_score(test_labels, test_predicts_proba)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUROC:    {auroc:.4f}")
    else:
        print("Test set contains only one class. Cannot compute AUROC.")

def main():
    cfg = OmegaConf.load(str(Path(__file__).resolve().parents[2] / "conf/custom_dreamt_chronos_finetune.yaml"))
    all_epochs_data = []
    all_epochs_labels = []

    df = pd.read_csv(cfg.data.path)

    first_wake_index = df[df['Sleep_Stage'] == 'W'].index[0]
    start_index = first_wake_index - (first_wake_index % SAMPLES_PER_EPOCH)
    df = df.iloc[start_index:].copy()

    df.reset_index(drop=True, inplace=True)
    df['Sleep_Stage'] = df['Sleep_Stage'].ffill()
    valid_stages = ['W', 'N1', 'N2', 'N3', 'R']
    df = df[df['Sleep_Stage'].isin(valid_stages)].copy()
    df['label'] = df['Sleep_Stage'].apply(lambda x: 0 if x == 'W' else 1)

    df['BVP_prep'] = preprocess_BVP(df['BVP'].to_numpy())
    df['ACC_X_prep'] = preprocess_ACC(df['ACC_X'].to_numpy())
    df['ACC_Y_prep'] = preprocess_ACC(df['ACC_Y'].to_numpy())
    df['ACC_Z_prep'] = preprocess_ACC(df['ACC_Z'].to_numpy())

    num_epochs = len(df) // SAMPLES_PER_EPOCH
    for i in tqdm(range(num_epochs), desc="Creating epochs"):
        start_idx = i * SAMPLES_PER_EPOCH
        end_idx = start_idx + SAMPLES_PER_EPOCH
        epoch_df = df.iloc[start_idx:end_idx]
        epoch_label = epoch_df['label'].iloc[0]
        epoch_channels = epoch_df[['BVP_prep', 'ACC_X_prep', 'ACC_Y_prep', 'ACC_Z_prep']].values
        all_epochs_data.append(epoch_channels)
        all_epochs_labels.append(epoch_label)

    X_all = np.array(all_epochs_data)
    y_all = np.array(all_epochs_labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    num_epochs, seq_len, num_channels = X_train.shape
    X_train_flattened = X_train.reshape(num_epochs, seq_len * num_channels)
    smote = SMOTE(random_state=42)
    X_train_resampled_flat, y_train_resampled = smote.fit_resample(X_train_flattened, y_train)
    X_train_resampled = X_train_resampled_flat.reshape(-1, seq_len, num_channels)

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ChronosEmbeddingDataset(X_train_resampled, y_train_resampled, pipeline)
    val_dataset = ChronosEmbeddingDataset(X_val, y_val, pipeline)
    test_dataset = ChronosEmbeddingDataset(X_test, y_test, pipeline)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        first_embedding, _ = train_dataset[0]
        input_dim = first_embedding.shape[0]

    model = ClassifierHead(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    NUM_EPOCHS = 15

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        train_labels = []
        train_predicts = []
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            inputs, labels = inputs.to(device), labels.unsqueeze(1).to(device)
            print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_labels.extend(labels.cpu().numpy())
            train_predicts.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_auroc = roc_auc_score(train_labels, train_predicts)
        
        model.eval()
        total_val_loss = 0
        val_labels = []
        val_predicts = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                val_labels.extend(labels.cpu().numpy())
                val_predicts.extend(torch.sigmoid(outputs).cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_auroc = roc_auc_score(val_labels, val_predicts) if len(np.unique(val_labels)) > 1 else 0.0

        print(f"Epoch {epoch+1:02}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train AUROC: {train_auroc:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUROC: {val_auroc:.4f}")

    if len(test_loader.dataset) > 0:
        test_model(model, test_loader, device)
    else:
        print("\nTest set is empty. Skipping final evaluation.")

if __name__ == "__main__":
    main()
