class ModelConfig:
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, 
                 feedforward_dim, dropout, layer_activation="gelu", seq_len=None, lr=1e-4, return_intermediate=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout
        self.layer_activation = layer_activation
        self.seq_len = seq_len
        self.lr = lr
        self.return_intermediate = return_intermediate