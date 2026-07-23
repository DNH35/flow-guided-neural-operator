try:
    import wandb
except ImportError:  # optional; pretrain logging only
    wandb = None

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import time
from models.transformer_encoder_input import TransformerEncoderInput
from models.spec_prediction_head import SpecPredictionHead
from models.model_config import ModelConfig
from models.util.util import make_grid, reshape_for_batchwise
from models.util.gaussian_process import GPPrior
from tqdm import tqdm
import matplotlib.pyplot as plt


def _wandb_log(payload, step=None):
    if wandb is None or wandb.run is None:
        return
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)

class TransformerModel(nn.Module):
    def __init__(self, cfg):
        super(TransformerModel, self).__init__()
        self.cfg = cfg
        self.encoder_input = TransformerEncoderInput(cfg, dropout=cfg.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation=cfg.layer_activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers
        )
        
        self.prediction_head = SpecPredictionHead(cfg)
    
    def forward(self, x, intermediate_rep=False):
        encoded_input, _ = self.encoder_input(x)
        transformer_output = self.transformer_encoder(encoded_input)
        if intermediate_rep:
            return transformer_output
        output = self.prediction_head(transformer_output)
        return output

class TransformerFFM(nn.Module):
    def __init__(self, cfg, intermediate_rep=False):
        super(TransformerFFM, self).__init__()
        
        # Create a modified config for the time-conditioned input
        self.time_cfg = ModelConfig(
            input_dim=cfg.input_dim + 1,  # +1 for time feature
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            feedforward_dim=cfg.feedforward_dim,
            dropout=cfg.dropout,
            layer_activation=cfg.layer_activation,
            seq_len=cfg.seq_len
        )
        self.intermediate_rep = intermediate_rep
        self.transformer = TransformerModel(self.time_cfg)
        original_input_dim = cfg.input_dim
        self.transformer.prediction_head.output = nn.Linear(
            self.time_cfg.hidden_dim, 
            original_input_dim
        )
        self.cfg = cfg
        self.cfg = cfg
    
    def forward(self, t, x):
        batch_size, seq_len, feature_dim = x.shape
        
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        if t.shape[0] == 1 and batch_size > 1:
            t = t.repeat(batch_size)
        
        t_expanded = t.view(batch_size, 1, 1).expand(batch_size, seq_len, 1)
        
        x_with_time = torch.cat([x, t_expanded], dim=-1)  # [batch_size, seq_len, feature_dim+1]
        output = self.transformer(x_with_time, self.intermediate_rep)
        
        return output

class NeuralFFMModel:
    def __init__(self, model, cfg, kernel_length=0.001, kernel_variance=1.0, sigma_min=1e-4, device='cpu', dtype=torch.float32, vp=False, ema_decay=0.999):
        self.model = model.to(device) 
        self.device = device
        self.dtype = dtype
        self.gp = GPPrior(lengthscale=kernel_length, var=kernel_variance, device=device)
        self.cfg = cfg
        self.sigma_min = sigma_min
        self.ema_params = None 
        self.ema_decay = ema_decay
        self.ema_decay = ema_decay
        self.vp = vp
        if self.vp:
            self.alpha, self.dalpha = self.construct_alpha()

    def construct_alpha(self):
        def alpha(t):
            return torch.cos((t + 0.08)/2.16 * np.pi).to(self.device)
        def dalpha(t):
            return -np.pi/2.16 * torch.sin((t + 0.08)/2.16 * np.pi).to(self.device)
        return alpha, dalpha
    
    def simulate(self, t, x_data):
        batch_size = x_data.shape[0]
        n_channels = x_data.shape[1]
        dims = x_data.shape[2:]
        n_dims = len(dims)
        
        query_points = make_grid(dims)
        noise = self.gp.sample(query_points, dims, n_samples=batch_size, n_channels=n_channels)

        t_reshaped = reshape_for_batchwise(t, 1 + n_dims)
        if self.vp:
            mu = self.alpha(1-t_reshaped) * x_data
            sigma = torch.sqrt((1 - self.alpha(1-t_reshaped)**2))
        else:
            
            mu = t_reshaped * x_data
            sigma = 1. - (1. - self.sigma_min) * t_reshaped

        samples = mu + sigma * noise

        assert samples.shape == x_data.shape
        return samples
    
    def get_conditional_fields(self, t, x_data, x_noisy):
        dims = x_data.shape[2:]
        n_dims = len(dims)
        t_reshaped = reshape_for_batchwise(t, 1 + n_dims)
        if self.vp:
            conditional_fields = (self.dalpha(1-t_reshaped)/(1 - self.alpha(1-t_reshaped)**2)) * (self.alpha(1-t_reshaped)*x_noisy - x_data)
        else:
            c = 1. - (1. - self.sigma_min) * t_reshaped
            conditional_fields = (x_data - (1. - self.sigma_min) * x_noisy) / c

        return conditional_fields

    def train(self, train_loader, optimizer, epochs, 
            scheduler=None, test_loader=None, eval_int=0, 
            save_int=0, generate=False, save_path=None, model_type=""):
        
        tr_losses = []
        te_losses = []
        eval_eps = []
        evaluate = (eval_int > 0) and (test_loader is not None)
        print("EVALUATE: ", evaluate)
        print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<12} | Time")
        print("-" * 45)
        model = self.model
        device = self.device
        dtype = self.dtype
        train_epoch_time = 0
        sample_size = len(train_loader.dataset)
        print(f"SAMPLE SIZE: {sample_size}")
        for ep in tqdm(range(1, epochs+1)):
            t0 = time.time()
            epoch_start = time.time()
            train_start = time.time()
            model.train()
            tr_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {ep} Training", leave=False)
            for batch_idx, batch_data in enumerate(train_pbar):
                train_batch_data = batch_data["target"]

                batch = train_batch_data
                    
                batch = batch.to(device)

                batch_size = batch.shape[0]

                t = torch.rand(batch_size, device=device)
                x_noisy = self.simulate(t, batch)
                
                target = self.get_conditional_fields(t, batch, x_noisy)
                model_out = model(t, x_noisy)

                optimizer.zero_grad()
                loss = torch.mean((model_out - target)**2)
                loss.backward()
                optimizer.step()

                if self.ema_params is None:
                    self.ema_params = {name: p.data.clone().detach() for name, p in model.named_parameters()}
                else:
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            self.ema_params[name] = self.ema_decay * self.ema_params[name] + (1 - self.ema_decay) * param.data
                tr_loss += loss.item()
            
            train_duration = time.time() - train_start

            tr_loss /= len(train_loader)
            tr_losses.append(tr_loss)
            _wandb_log({"train_loss": tr_loss}, step=ep)
            if scheduler: scheduler.step()

            t1 = time.time()
            epoch_time = t1 - t0
            print(f'tr @ epoch {ep}/{epochs} | Loss {tr_loss:.6f} | {epoch_time:.2f} (s)')
 
            if generate:
                with torch.no_grad():
                    model.eval()
                    real_sample = batch[0].cpu().numpy()
                   
                    n_channels= batch.shape[1]

                    generated_samples = self.sample(
                        dims=batch.shape[2:], 
                        batch_size=batch_size, 
                        seq_len=n_channels,
                        feature_dim=self.cfg.input_dim,
                    )
                    self.plot_spectrograms(real_sample, generated_samples[0].cpu().numpy(), ep, save_path)

            if eval_int > 0:
                t0 = time.time()
                eval_eps.append(ep)

                with torch.no_grad():
                    model.eval()
                    evaluate_start = time.time()
                    if evaluate:
                        te_loss = 0.0
                        original_state = model.state_dict()

                        if self.ema_params is not None:
                            ema_state_dict = model.state_dict()
                            for name, param in self.ema_params.items():
                                ema_state_dict[name] = param
                            model.load_state_dict(ema_state_dict, strict=False)

                        val_pbar = tqdm(test_loader, desc=f"Epoch {ep} Validation", leave=False)
                        for batch_data in val_pbar:
                            val_batch_data = batch_data["target"]
                            batch = val_batch_data
                            batch = batch.to(device)
                            batch_size = batch.shape[0]

                            t = torch.rand(batch_size, device=device)
                            
                            x_noisy = self.simulate(t, batch)
                            
                            target = self.get_conditional_fields(t, batch, x_noisy)

                            model_out = model(t, x_noisy)

                            loss = torch.mean((model_out - target)**2)
                            te_loss += loss.item()

                        te_loss /= len(test_loader)
                        te_losses.append(te_loss)
                        model.load_state_dict(original_state, strict=False)
                        _wandb_log({"val_loss": te_loss}, step=ep)

                        
                        t1 = time.time()
                        epoch_time = t1 - t0
                        print(f'te @ epoch {ep}/{epochs} | Loss {te_loss:.6f} | {epoch_time:.2f} (s)')
                        evaluate_duration = time.time() - evaluate_start
                    
            plt.plot(tr_losses, label='Train Loss')
            plt.plot(te_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Val Losses')
            plt.legend()
            
            if ep % 1 == 0:
                saved_model = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hidden_dim': self.cfg.hidden_dim,
                }
                save_start_time = time.time()
                torch.save(saved_model, save_path / f'{model_type}.pt')
                save_end_time = time.time() - save_start_time

            pipeline_duration = time.time() - epoch_start
            _wandb_log({
                "train_epoch_time": train_duration,
                "evaluate_epoch_time": evaluate_duration if evaluate else 0,
                "save_epoch_time": save_end_time if ep % 10 == 0 else 0,
                "pipeline_epoch_time": pipeline_duration,

            }, step=ep)
    @torch.no_grad()
    def sample(self, dims, batch_size=1, seq_len=None, feature_dim=None, n_eval=2, return_path=False, rtol=1e-5, atol=1e-5):
        """
            Generated samples of shape [batch_size, seq_len, feature_dim]
        """
        if seq_len is None:
            seq_len = self.cfg.seq_len
        if feature_dim is None:
            feature_dim = self.cfg.input_dim
        
        t = torch.linspace(0, 1, n_eval, device=self.device)
        grid = make_grid(dims)
        x0 = self.gp.sample(grid, dims, n_channels=seq_len, n_samples=batch_size)
        method = 'dopri5'
        out = odeint(self.model, x0, t, method=method, rtol=rtol, atol=atol)

        if return_path:
            return out
        else:
            return out[-1]
    def plot_sample_stft(self, sample, fs, nperseg, noverlap, freq_cutoff):
        hop_length = nperseg - noverlap
        num_frames, num_freq_bins = sample.shape
        time_bins = np.arange(num_frames + 1) * hop_length / fs
        freq_resolution = fs / nperseg
        max_bin = int(freq_cutoff / freq_resolution)

        magnitude = sample[:, :max_bin]
        freq_edges = np.linspace(0, freq_cutoff, max_bin + 1)

        time_grid, freq_grid = np.meshgrid(time_bins, freq_edges, indexing='ij')
        plt.figure(figsize=(15, 6))
        plt.pcolormesh(time_grid, freq_grid, magnitude.T, shading='gouraud', vmin=-3, vmax=5)
        x = np.percentile(magnitude, 99)
        plt.colorbar(label="Power (Arbitrary units)")
        plt.title("Generated Sample", fontsize=14)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Frequency (Hz)", fontsize=12)
        plt.ylim(0, freq_cutoff)
        plt.show()
        
    def plot_spectrograms(self, real_spec, generated_spec, epoch, plot_dir):
        fig = plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        self._plot_single_spec(real_spec, "Real Sample")
        
        plt.subplot(1, 2, 2)
        self._plot_single_spec(generated_spec, "Generated Sample")
        
        plt.suptitle(f"Epoch {epoch}", fontsize=16)
        plt.tight_layout()
        if wandb is not None:
            _wandb_log({"real_vs_generated_spectrograms": wandb.Image(fig)}, step=epoch)
        plt.close(fig)

    def _plot_single_spec(self, spec, title):
        duration_seconds = 3.0
        sr = 400  
        
        n_time_bins = spec.shape[0]
        t = np.linspace(0, duration_seconds, n_time_bins)
        
        f = np.linspace(0, 40, spec.shape[1])  
        
        plt.pcolormesh(t, f, spec.T, shading='gouraud', vmin=-3, vmax=5)
        plt.colorbar(label="Power (Arbitrary units)")
        plt.title(title, fontsize=14)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Frequency (Hz)", fontsize=12)
        plt.xlim(0, duration_seconds)  


    def plot_stft(self, wav):
        f,t,linear = self.get_stft(wav, 2048, clip_fs=40, nperseg=400, noverlap=350, normalizing="zscore", return_onesided=True) #TODO hardcode sampling rate
        plt.figure(figsize=(15,3))
        f[-1]=200
        g1 = plt.pcolormesh(t,f,linear, shading="gouraud", vmin=-3, vmax=5)
        print(t.shape)
        print(f.shape)
        print(linear.shape)
        cbar = plt.colorbar(g1)
        tick_font_size = 15
        cbar.ax.tick_params(labelsize=tick_font_size)
        cbar.ax.set_ylabel("Power (Arbitrary units)", fontsize=15)
        plt.title("REAL SAMPLE", fontsize=15)
        plt.xticks(fontsize=20)
        plt.ylabel("")
        plt.yticks(fontsize=20)
        plt.xlabel("Time (s)", fontsize=20)
        plt.ylabel("Frequency (Hz)", fontsize=20)

