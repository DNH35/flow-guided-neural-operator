from torch.utils.data import Dataset
from scipy.signal import decimate
import torch
import numpy as np

class DownsampledWavsDataset(Dataset):
    def __init__(self, wavs_data, labels_data, downsample_factor=1):
        if len(wavs_data) != len(labels_data):
            raise ValueError("The number of wavs samples and labels must be the same.")

        self.wavs_data = np.array(wavs_data)
        self.labels_data = np.array(labels_data)
        self.downsample_factor = downsample_factor

    def __len__(self):
        return len(self.labels_data)

    def __getitem__(self, idx):
        wav_sample = self.wavs_data[idx]
        label_sample = self.labels_data[idx]

        if self.downsample_factor > 1:
            downsampled_wav = decimate(wav_sample, self.downsample_factor)  
        else:
            downsampled_wav = wav_sample

        wav_tensor = torch.tensor(downsampled_wav, dtype=torch.float32)
        label_tensor = torch.tensor(label_sample, dtype=torch.float32)

        return {'wavs': wav_tensor, 'labels': label_tensor}


class WavsChronosEmbeddingDataset(Dataset):
    def __init__(self, raw_wavs_data, labels_data, chronos_pipeline, downsample_factor=1):
        if len(raw_wavs_data) != len(labels_data):
            raise ValueError("The number of wavs samples and labels must be the same.")

        self.raw_wavs_data = np.array(raw_wavs_data)
        self.labels_data = np.array(labels_data)
        self.pipeline = chronos_pipeline
        self.downsample_factor = downsample_factor

    def __len__(self):
        return len(self.labels_data)

    def __getitem__(self, idx):
        wav_sample = self.raw_wavs_data[idx]
        label_sample = self.labels_data[idx]

        if self.downsample_factor > 1:
            downsampled_wav = wav_sample[::self.downsample_factor]
        else:
            downsampled_wav = wav_sample

        context = torch.tensor(downsampled_wav, dtype=torch.bfloat16).unsqueeze(0)

        with torch.no_grad():
            embeddings_per_channel, _ = self.pipeline.embed(context)

        final_embedding = embeddings_per_channel[:, -1, :].mean(axis=0)
        detached_embedding = final_embedding.detach().to(torch.float32)

        return detached_embedding, torch.tensor(label_sample, dtype=torch.float32)
    