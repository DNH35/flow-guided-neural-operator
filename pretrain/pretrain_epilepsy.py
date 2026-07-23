"""Pretrain the FFM backbone on epilepsy STFT windows.

Expects data/epilepsy/{train,val,test}_stft.pt (see data_preprocess/epilepsy).
Writes the pretrained checkpoint to checkpoints/ffm_epilepsy_pretrain.pt,
which is what conf/finetune_epilepsy*.yaml expect as model.upstream_ckpt.
"""
import torch
import torch_optimizer as optim
from torch.utils.data import DataLoader

from data.UniMib_data.dataset.dataset import PretrainUniMibDataset
from models.ffm_transformer import TransformerFFM, NeuralFFMModel
from models.model_config import ModelConfig
from pretrain_utils import fgno_root, maybe_init_wandb

BATCH_SIZE = 64
EPOCHS = 250
MODEL_NAME = "ffm_epilepsy_pretrain"

if __name__ == "__main__":
    ROOT = fgno_root()
    DATA_DIR = ROOT / "data" / "epilepsy"
    SAVE_DIR = ROOT / "checkpoints"
    SAVE_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Determining data dimensions for ModelConfig...")
    temp_data = torch.load(DATA_DIR / "train_stft.pt")["samples"]
    # Shape is (N, TimeSteps, FeatureDim)
    seq_len = temp_data.shape[1]
    input_dim = temp_data.shape[2]
    print(f"Detected Data Shape: sequence_length={seq_len}, input_dim={input_dim}")

    cfg = ModelConfig(
        input_dim=input_dim,
        hidden_dim=768,
        num_heads=12,
        num_layers=6,
        feedforward_dim=3072,
        dropout=0.1,
        seq_len=seq_len,
        lr=0.0001,
    )

    print("\nCreating datasets and dataloaders...")
    train_dataset = PretrainUniMibDataset(DATA_DIR / "train_stft.pt")
    val_dataset = PretrainUniMibDataset(DATA_DIR / "val_stft.pt")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print("DataLoaders created successfully.")

    model = TransformerFFM(cfg)
    ffm_model = NeuralFFMModel(model, cfg, kernel_length=0.001, kernel_variance=1.0, device=device)
    optimizer = optim.Lamb(ffm_model.model.parameters(), lr=cfg.lr)

    maybe_init_wandb(project="flow_matching_unimib_shar", run_name=MODEL_NAME)

    print("\nStarting pretraining...")
    ffm_model.train(
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=EPOCHS,
        scheduler=None,
        test_loader=val_loader,
        eval_int=2,
        save_path=SAVE_DIR,
        model_type=MODEL_NAME,
    )
    print(f"Training finished. Checkpoint saved to {SAVE_DIR / (MODEL_NAME + '.pt')}")
