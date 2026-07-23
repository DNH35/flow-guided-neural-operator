from models.ffm_transformer import TransformerModel
from models.model_config import ModelConfig
import torch
import torch.nn as nn

class MaskedAutoencoderTransformer(nn.Module):
    """
    The Transformer-based MAE model architecture used for pre-training.
    We need this definition to load the saved model weights.
    """
    def __init__(self, cfg: ModelConfig):
        super(MaskedAutoencoderTransformer, self).__init__()
        self.cfg = cfg
        self.transformer = TransformerModel(self.cfg)

    def forward(self, masked_input: torch.Tensor) -> torch.Tensor:
        return self.transformer(
            masked_input,
            intermediate_rep=getattr(self.cfg, "return_intermediate", False),
        )