# from models import register_model
# import torch.nn as nn
# import torch
# from models.base_model import BaseModel
# from models.transformer_encoder_input import TransformerEncoderInput

# @register_model("finetune_model")
# class FinetuneModel(BaseModel):
#     def __init__(self):
#         super(FinetuneModel, self).__init__()

#     def forward(self, inputs, pad_mask):
#         if self.frozen_upstream:
#             self.upstream.eval()
#             with torch.no_grad():
#                 outputs = self.upstream(inputs, pad_mask, intermediate_rep=True)
#         else:
#             outputs = self.upstream(inputs, pad_mask, intermediate_rep=True)
#         middle = int(outputs.shape[1]/2)
#         outputs = outputs[:,middle-5:middle+5].mean(axis=1)
#         out = self.linear_out(outputs)
#         return out

#     def build_model(self, cfg, upstream_model):
#         self.cfg = cfg
#         self.upstream = upstream_model
#         self.upstream_cfg = self.upstream.cfg
#         hidden_dim = self.upstream_cfg.hidden_dim
#         self.linear_out = nn.Linear(in_features=hidden_dim, out_features=1) #TODO hardcode out_features
#         self.frozen_upstream = cfg.frozen_upstream

from models import register_model
import torch.nn as nn
import torch
from models.base_model import BaseModel
import logging

log = logging.getLogger(__name__)

# @register_model("finetune_model")
# class FinetuneModel(BaseModel):
#     def __init__(self):
#         super(FinetuneModel, self).__init__()

#     def forward(self, inputs, pad_mask):
#         if self.entire_backbone_frozen:
#             self.upstream.eval()
#             with torch.no_grad():
#                 outputs = self.upstream(inputs, pad_mask, intermediate_rep=True)
#         else:
#             outputs = self.upstream(inputs, pad_mask, intermediate_rep=True)
        
#         # Original temporal pooling
#         middle = int(outputs.shape[1]/2)
#         # outputs = outputs.mean(dim=1)  # Global pooling
#         outputs = outputs[:,middle-5:middle+5].mean(axis=1)

        
#         # Pass through multi-layer classifier head
#         out = self.classifier_head(outputs)
#         return out.squeeze(-1)

#     def build_model(self, cfg, upstream_model):
#         self.cfg = cfg
#         self.upstream = upstream_model
#         self.upstream_cfg = self.upstream.cfg
#         hidden_dim = self.upstream_cfg.hidden_dim
        
#         # Get freeze strategy from config (default to "all")
#         freeze_strategy = cfg.get("freeze_strategy", "all")
        
#         # Freeze entire backbone by default
#         for param in self.upstream.parameters():
#             param.requires_grad = False
            
#         # Apply layer unfreezing strategy
#         encoder_layers = self._get_encoder_layers()
#         num_layers = len(encoder_layers) if encoder_layers else 0
        
#         if freeze_strategy == "none":
#             # Unfreeze entire backbone
#             for param in self.upstream.parameters():
#                 param.requires_grad = True
                
#         elif freeze_strategy.startswith("last_"):
#             k = int(freeze_strategy.split("_")[1])
#             for i in range(num_layers - k, num_layers):
#                 for param in encoder_layers[i].parameters():
#                     param.requires_grad = True
                    
#         elif freeze_strategy.startswith("first_"):
#             k = int(freeze_strategy.split("_")[1])
#             for i in range(k):
#                 for param in encoder_layers[i].parameters():
#                     param.requires_grad = True
                    
#         elif freeze_strategy.startswith("middle_"):
#             k = int(freeze_strategy.split("_")[1])
#             start = (num_layers - k) // 2
#             for i in range(start, start + k):
#                 for param in encoder_layers[i].parameters():
#                     param.requires_grad = True
        
#         # Check if backbone is completely frozen
#         self.entire_backbone_frozen = not any(
#             p.requires_grad for p in self.upstream.parameters()
#         )
        
#         # Build multi-layer classifier head (same as your model)
#         # self.classifier_head = nn.Sequential(
#         #     nn.Linear(hidden_dim, hidden_dim // 2),
#         #     nn.ReLU(),
#         #     nn.Linear(hidden_dim // 2, hidden_dim // 4),
#         #     nn.ReLU(),
#         #     nn.Linear(hidden_dim // 4, 1)
#         # )
#         self.classifier_head = nn.Linear(in_features=hidden_dim, out_features=1)
        
#         # Print trainable parameters
#         trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         total_params = sum(p.numel() for p in self.parameters())
#         log.info(f"Freeze strategy: {freeze_strategy}")
#         log.info(f"Trainable params: {trainable_params}/{total_params} ({trainable_params/total_params:.2%})")
        
#     def _get_encoder_layers(self):
#         """Helper to get encoder layers from different upstream structures"""
#         # Try different possible locations for encoder layers
#         if hasattr(self.upstream, 'transformer_encoder') and hasattr(self.upstream.transformer_encoder, 'layers'):
#             return self.upstream.transformer_encoder.layers
#         elif hasattr(self.upstream, 'encoder') and hasattr(self.upstream.encoder, 'layers'):
#             return self.upstream.encoder.layers
#         elif hasattr(self.upstream, 'transformer') and hasattr(self.upstream.transformer, 'layers'):
#             return self.upstream.transformer.layers
#         return None 

import torch
import torch.nn as nn
from models.base_model import BaseModel
from models import register_model

@register_model("finetune_model")
class FinetuneModel(BaseModel):
    """
    A fine-tuning model that extracts features from specified intermediate
    layers of a frozen upstream transformer, concatenates them, and passes
    them through a linear classifier.
    """
    def __init__(self):
        super(FinetuneModel, self).__init__()
        self.features_container = []  # Stores intermediate features during the forward pass
        self.hook_handles = []       # Stores hook handles for later removal

    def _hook_fn(self, module, input, output):
        """
        Hook function to capture the output of an intermediate layer.
        It assumes the relevant feature tensor is the first element of the output tuple,
        which is common in transformer implementations.
        """
        # Transformer layer output is often a tuple (hidden_states, attention_weights)
        feature = output[0] if isinstance(output, tuple) else output
        self.features_container.append(feature)

    def build_model(self, cfg, upstream_model):
        """
        Builds the fine-tuning model by attaching hooks to the upstream model
        and initializing the final classifier head.
        """
        self.cfg = cfg
        self.upstream = upstream_model
        self.upstream_cfg = self.upstream.cfg
        self.frozen_upstream = cfg.frozen_upstream

        # Freeze the entire upstream model
        if self.frozen_upstream:
            for param in self.upstream.parameters():
                param.requires_grad = False
            self.upstream.eval()

        # Access the encoder layers of the upstream transformer
        # Note: The exact path may need adjustment depending on the upstream model's architecture
        encoder_layers = self.upstream.transformer.layers
        
        # Determine which layers to extract features from
        # Expects a list of integers in the config, e.g., `feature_layers: [0, 2, 5]`]
        layer_indices = self.cfg.get("feature_layers", [-1]) # Default to the last layer
        if layer_indices == [-1]:
            layer_indices = [len(encoder_layers) - 1] # Use the last layer if not specified
        
        print(f"Registering forward hooks on layers: {layer_indices}")

        # Register a forward hook for each specified layer
        for idx in layer_indices:
            if idx < len(encoder_layers):
                handle = encoder_layers[idx].register_forward_hook(self._hook_fn)
                self.hook_handles.append(handle)
            else:
                print(f"Warning: Layer index {idx} is out of bounds.")

        # Dynamically define the classifier based on the number of hooked layers
        hidden_dim = self.upstream_cfg.hidden_dim
        num_feature_layers = len(self.hook_handles)
        classifier_input_dim = num_feature_layers * hidden_dim
        
        # Define the output classification head
        self.linear_out = nn.Linear(in_features=classifier_input_dim, out_features=1)
        
        print(f"Classifier head initialized with input dimension: {classifier_input_dim}")

    def forward(self, inputs, pad_mask):
        """
        Defines the forward pass of the model.
        """
        # 1. Reset the feature container for the new forward pass
        self.features_container = []

        # 2. Run the forward pass on the (potentially frozen) upstream model
        # The hooks will automatically capture features from the specified layers
        if self.frozen_upstream:
            with torch.no_grad():
                _ = self.upstream(inputs, pad_mask, intermediate_rep=True)
        else:
            _ = self.upstream(inputs, pad_mask, intermediate_rep=True)

        # 3. Process the captured features
        # Apply global average pooling to each captured feature tensor
        pooled_features = [f.mean(dim=0) for f in self.features_container]
        # Concatenate the pooled features along the feature dimension
        combined_features = torch.cat(pooled_features, dim=1)

        # 4. Pass the combined features through the final classifier
        out = self.linear_out(combined_features)
        return out.squeeze(-1)

    def __del__(self):
        """
        Destructor to ensure that all registered hooks are removed when the
        model object is destroyed, preventing potential memory leaks.
        """
        for handle in self.hook_handles:
            handle.remove()
            #rsync -avz --exclude '/home/duy/TrajectoryFlowMatching/BrainBERT/all_day_data' --exclude '/home/duy/TrajectoryFlowMatching/BrainBERT/braintreebank_data' /home/duy/TrajectoryFlowMatching/BrainBERT duy@131.215.142.243:/home/duy/BrainBERT
            #rsync -avz --exclude '/home/duy/TrajectoryFlowMatching/BrainBERT/all_day_data' --exclude '/home/duy/TrajectoryFlowMatching/BrainBERT/braintreebank_data' rsync -avz --exclude={'all_day_data','braintreebank_data'} duy@131.215.142.161:/home/duy/TrajectoryFlowMatching/BrainBERT /home/duynguyen /home/dhnguyen