from scipy import signal
import numpy as np
import torch
import torch.nn as nn

def _first(arr, axis):
    """Return arr[..., 0:1, ...] where 0:1 is in the `axis` position."""
    return np.take_along_axis(arr, np.array(0, ndmin=arr.ndim), axis)

def zscore(a, axis):
    mn = a.mean(axis=axis, keepdims=True)
    std = a.std(axis=axis, ddof=0, keepdims=True)

    std[(std==0)] = 1.0 
    z = (a - mn) / std
    return z
        
class STFTPreprocessor(nn.Module):
    def get_stft(self, x, fs, show_fs=-1, normalizing=None, **kwargs):
        if x.ndim > 1: x = x.squeeze()
        f, t, Zxx = signal.stft(x, fs, **kwargs)
        if "return_onesided" in kwargs and kwargs["return_onesided"] is True:
            if show_fs != -1: Zxx = Zxx[:show_fs]; f = f[:show_fs]
        Zxx = np.abs(Zxx)
        if normalizing == "zscore":
            Zxx = zscore(Zxx, axis=-1)
            if (Zxx.std() == 0).any(): Zxx = np.ones_like(Zxx)
        elif normalizing == "db":
            Zxx = np.log(Zxx + 1e-10)
        if np.isnan(Zxx).any(): Zxx = np.nan_to_num(Zxx, nan=0.0)
        return f, t, torch.Tensor(np.transpose(Zxx))

    def __init__(self, cfg, sampling_rate):
        super(STFTPreprocessor, self).__init__()
        self.cfg = cfg
        self.sampling_rate = sampling_rate

    def forward(self, wav: np.ndarray):
        all_channels_stft = []
        
        for i in range(wav.shape[-1]):
            channel_signal = wav[:, i]
            _, _, linear = self.get_stft(
                channel_signal, self.sampling_rate, show_fs=self.cfg.freq_channel_cutoff,
                nperseg=self.cfg.nperseg, noverlap=self.cfg.noverlap,
                normalizing=self.cfg.normalizing, return_onesided=True
            )
            all_channels_stft.append(linear)
        
        return torch.cat(all_channels_stft, dim=1)