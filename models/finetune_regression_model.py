import torch
import torch.nn as nn
class FinetuneRegressionModel(nn.Module):
    def __init__(self, feature_extraction_time, use_pooling=False):
        super().__init__()
        self.feature_extraction_time = feature_extraction_time
        self.features_container = []
        self.hook_handles = []
        self.use_pooling = use_pooling

    def forward(self, inputs):
        t = torch.full((inputs.shape[0],), self.feature_extraction_time, device=inputs.device)

        self.features_container = [] 

        x_noisy = self.model.simulate(t, inputs)

        with torch.no_grad():
            _ = self.model.model(t, inputs)

        processed_features = []
        for f in self.features_container:            
            if self.use_pooling:
                feat = f.mean(dim=1)
            else:
                feat = torch.flatten(f, start_dim=1)
            
            processed_features.append(feat)

        combined_features = torch.cat(processed_features, dim=1)
        out = self.linear_out(combined_features)

        return out.squeeze(-1)

    def _hook_fn(self, module, input, output):
        feature = output[0] if isinstance(output, tuple) else output
        self.features_container.append(feature)

    def build_model(self, cfg, load_model, hidden_dim, seq_len, device, layer_indices):
        self.cfg = cfg
        self.model = load_model

        for param in self.model.model.parameters():
            param.requires_grad = False
        
        self.model.model.eval()

        encoder_layers = self.model.model.transformer.transformer_encoder.layers
        for idx in layer_indices:
            handle = encoder_layers[idx].register_forward_hook(self._hook_fn)
            self.hook_handles.append(handle)

        if self.use_pooling:
            input_features = hidden_dim
        else:
            input_features = seq_len * hidden_dim
            
        head_input_dim = len(layer_indices) * input_features

        self.linear_out = nn.Sequential(
            nn.BatchNorm1d(head_input_dim),
            nn.Dropout(p=0.2),
            nn.Linear(head_input_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, 1)
        ).to(device)

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()