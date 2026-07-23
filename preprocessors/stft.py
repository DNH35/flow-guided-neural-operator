import numpy as np
import torch
import torch.nn as nn
from scipy import signal, stats

def _first(arr, axis):
    #from https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_stats_py.py#L2662-L2730
    """Return arr[..., 0:1, ...] where 0:1 is in the `axis` position."""
    return np.take_along_axis(arr, np.array(0, ndmin=arr.ndim), axis)

def zscore(a, axis):
    #from https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_stats_py.py#L2662-L2730
    mn = a.mean(axis=axis, keepdims=True)
    std = a.std(axis=axis, ddof=0, keepdims=True)

    std[(std==0)] = 1.0
    z = (a - mn) / std
    return z

class STFTPreprocessor(nn.Module):
    def get_stft(self, x, fs, freq_cutoff, normalizing=None, **kwargs):
        if x.ndim == 1:
            x = x[np.newaxis, :]

        f, t, Zxx = signal.stft(x, fs, **kwargs)
        freq_cutoff_index = 40 
        if freq_cutoff_index == 0 and len(f) > 0:
            freq_cutoff_index = 1
        
        f = f[:freq_cutoff_index]
        Zxx = Zxx[:, :freq_cutoff_index, :]
        
        Zxx = np.abs(Zxx)

        if normalizing == "zscore" and Zxx.shape[-1] > 20:
            Zxx = zscore(Zxx, axis=-1)
            Zxx = Zxx[..., 10:-10] 
            t = t[10:-10]
        elif normalizing == "db":
            Zxx = np.log(Zxx + 1e-9)

        if np.isnan(Zxx).any():
            Zxx = np.nan_to_num(Zxx, nan=0.0)
        return f, t, Zxx.squeeze()

    def __init__(self, cfg, downsample_factor=1):
        super().__init__()
        self.cfg = cfg
        self.downsample_factor = self.cfg.downsample_factor
        if self.downsample_factor < 1:
            raise ValueError("Downsample factor must be >= 1.")
        self.original_fs = 2048

        print(f"DOWNSAMPLE FACTOR: {self.downsample_factor}")

    def forward(self, wav):
        if self.downsample_factor > 1:
            new_length = int(wav.shape[-1] / self.downsample_factor)
            downsampled_wav = signal.resample(wav, new_length, axis=-1)
        else:
            downsampled_wav = wav

        new_fs = self.original_fs / self.downsample_factor
        new_nperseg = int(self.cfg.nperseg / self.downsample_factor)
        new_noverlap = int(self.cfg.noverlap / self.downsample_factor)

        if new_nperseg < 1: new_nperseg = 1
        if new_noverlap >= new_nperseg: new_noverlap = new_nperseg - 1
        
        nyquist = new_fs / 2


        f, t, Zxx_processed = self.get_stft(
            downsampled_wav, fs=new_fs, freq_cutoff=self.cfg.freq_channel_cutoff, 
            normalizing=self.cfg.normalizing, return_onesided=True,
            nperseg=new_nperseg, noverlap=new_noverlap
        )
        
        target_bins = 40 #HARD CODED CHANGE THIS LATER
        formatting_mode = 'interpolate' 
        
        if self.downsample_factor > (self.original_fs / (2 * self.cfg.freq_channel_cutoff)):
            formatting_mode = 'pad_zeros'

        if Zxx_processed.ndim < 2 or Zxx_processed.size == 0:
            time_dim = len(t) if 't' in locals() and len(t) > 0 else 100
            Zxx_final = np.zeros((target_bins, time_dim))
        
        elif formatting_mode == 'pad_zeros':
            current_bins = Zxx_processed.shape[0]
            time_steps = Zxx_processed.shape[1]
            Zxx_final = np.zeros((target_bins, time_steps), dtype=Zxx_processed.dtype)
            bins_to_copy = min(current_bins, target_bins)
            Zxx_final[:bins_to_copy, :] = Zxx_processed[:bins_to_copy, :]
    
        else:
            Zxx_final = Zxx_processed
            if Zxx_final.shape[1] != target_bins:
                current_bins = Zxx_processed.shape[0]
                time_steps = Zxx_processed.shape[1]
                Zxx_final = np.zeros((target_bins, time_steps), dtype=Zxx_processed.dtype)
                bins_to_copy = min(current_bins, target_bins)
                Zxx_final[:bins_to_copy, :] = Zxx_processed[:bins_to_copy, :]

        return torch.Tensor(Zxx_final.T)