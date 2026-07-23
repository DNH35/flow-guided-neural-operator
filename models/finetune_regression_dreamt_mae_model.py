import torch
import torch.nn as nn

class FinetuneRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_container = []  
        self.hook_handles = []       
        self.backbone = None          
        self.linear_out = None       

    def _hook_fn(self, module, input, output):
        self.features_container.append(output)
    
    def build_model(self, cfg, backbone_model, hidden_dim, seq_len, device, layer_indices):
        self.backbone = backbone_model
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        encoder_layers = self.backbone.transformer.transformer_encoder.layers
        
        for idx in layer_indices:
            if idx < len(encoder_layers):
                handle = encoder_layers[idx].register_forward_hook(self._hook_fn)
                self.hook_handles.append(handle)
        
        head_input_dim = len(layer_indices) * seq_len * hidden_dim
        dropout_rate = 0.3

        self.linear_out = nn.Sequential(
            nn.BatchNorm1d(head_input_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(head_input_dim, head_input_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(head_input_dim // 4),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(head_input_dim // 4, 1)
        ).to(device)

    def forward(self, inputs):
        self.features_container = []
        
        with torch.no_grad():
            _ = self.backbone.transformer(inputs, intermediate_rep=True)

        flattened_features = [torch.flatten(f, start_dim=1) for f in self.features_container]
        combined_features = torch.cat(flattened_features, dim=1)
        
        out = self.linear_out(combined_features)
        return out.squeeze(-1)
    
    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()