import torch

from torch.utils.data import Dataset

class PretrainUniMibDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.samples = data["samples"].float()
        print(f"Loaded dataset from {data_path} with shape: {self.samples.shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"target": self.samples[idx]}