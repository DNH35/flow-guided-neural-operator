import torch
import torch.nn as nn
class FinetuneModel(nn.Module):
    def __init__(self, feature_extraction_time, noisy_input):
        super().__init__()
        self.feature_extraction_time = feature_extraction_time
        self.features_container = []
        self.hook_handles = []
        self.noisy_input = noisy_input

    def forward(self, inputs):
        self.features_container = []
        t = torch.full((inputs.shape[0],), self.feature_extraction_time, device=inputs.device)
        
        with torch.no_grad():
            if self.noisy_input:
                x_noisy = self.model.simulate(t, inputs)
                _ = self.model.model(t, x_noisy)
            else:
                _ = self.model.model(t, inputs)

        pooled = []
        for f in self.features_container:
            seq_len = f.shape[1]
            mid = seq_len // 2
            start = max(0, mid - 5)
            end = min(seq_len, mid + 5)
            pooled.append(f[:, start:end, :].mean(dim=1))  
            
        combined = torch.cat(pooled, dim=1) 
        out = self.linear_out(combined)
        return out

    def _hook_fn(self, module, input, output):
        feature = output[0] if isinstance(output, tuple) else output
        self.features_container.append(feature)

    def build_model(self, cfg, load_model, hidden_dim, device, num_classes, layer_indices):
        self.cfg = cfg
        self.model = load_model

        for param in self.model.model.parameters():
            param.requires_grad = False

        transformer = self.model.model.transformer
        encoder_layers = transformer.transformer_encoder.layers

        for idx in layer_indices:
            handle = encoder_layers[idx].register_forward_hook(self._hook_fn)
            self.hook_handles.append(handle)

        head_input_dim = len(layer_indices) * hidden_dim
        dropout_rate = 0.1
        h = head_input_dim

        self.linear_out = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, max(h // 2, 256)),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(max(h // 2, 256), num_classes)
        ).to(device)

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()