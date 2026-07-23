from torch.utils.data import Dataset
import torch

class FinetuneEpilepsyDataset(Dataset):
    """
    Dataset for unsupervised pretraining on the UniMib SHAR STFT data.
    It loads the pre-processed .pt file and returns only the samples, as
    labels are not needed for this pretraining task.
    """
    def __init__(self, data_path):
        data = torch.load(data_path)
        # We only need the samples for unsupervised pre-training
        self.samples = data["samples"].float()
        self.labels = data["labels"].float()
        print(f"Loaded dataset from {data_path} with shape: {self.samples.shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # The training loop expects the raw tensor, not a dictionary
        return self.samples[idx], self.labels[idx]