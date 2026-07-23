import torch
import torch.nn as nn

from models.ema_model import ModelEMA

__all__ = ["FinetuneFFMModel", "ModelEMA"]


class FinetuneFFMModel(torch.nn.Module):
    """Linear probe on frozen FFM transformer layer features at a fixed flow time."""

    def __init__(self, feature_extraction_time):
        super().__init__()
        self.feature_extraction_time = feature_extraction_time
        self.features_container = []
        self.hook_handles = []

    def forward(self, inputs):
        t = torch.full((inputs.shape[0],), self.feature_extraction_time, device=inputs.device)
        self.features_container = []

        with torch.no_grad():
            _ = self.model.model(t, inputs)

        pooled_features = [f.mean(dim=1) for f in self.features_container]
        combined_features = torch.cat(pooled_features, dim=1)
        out = self.linear_out(combined_features)
        return out.squeeze(-1)

    def _hook_fn(self, module, input, output):
        feature = output[0] if isinstance(output, tuple) else output
        self.features_container.append(feature)

    def build_model(self, cfg, load_model, hidden_dim, device, layer_indices=-1, dropout_p=0.1):
        self.cfg = cfg
        self.model = load_model

        for param in self.model.model.parameters():
            param.requires_grad = False

        transformer = self.model.model.transformer
        encoder_layers = transformer.transformer_encoder.layers
        num_layers = len(encoder_layers)

        if isinstance(layer_indices, int):
            layer_indices = [layer_indices] if layer_indices != -1 else [num_layers - 1]

        for idx in layer_indices:
            handle = encoder_layers[idx].register_forward_hook(self._hook_fn)
            self.hook_handles.append(handle)

        head_input_dim = len(layer_indices) * hidden_dim
        self.linear_out = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(head_input_dim // 2, head_input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(head_input_dim // 4, 1),
        ).to(device)

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()
