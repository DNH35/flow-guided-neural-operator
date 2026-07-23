import torch
from torch.utils.data import Dataset

class FinetuneUniMibDataset(Dataset):
    def __init__(self, data_path, num_classes=5):
        data = torch.load(data_path)
        self.samples = data["samples"].float()
        self.labels = data["labels"].float()
        self.num_classes = num_classes
        print(f"Loaded dataset from {data_path} with shape: {self.samples.shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, label = self.samples[idx], self.labels[idx]
        return sample, label