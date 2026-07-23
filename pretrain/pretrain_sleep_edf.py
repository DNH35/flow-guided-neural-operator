"""Pretrain the FFM backbone on Sleep-EDF STFT windows.

Expects data/sleepEDF/{train,val,test}_stft.pt (see data_preprocess/sleep_edf).
Writes the pretrained checkpoint to checkpoints/ffm_sleepEDF_pretrain.pt,
which is what conf/finetune_sleepEDF*.yaml expect as model.upstream_ckpt.
"""
import torch
import torch_optimizer as optim
from torch.utils.data import DataLoader

from data.UniMib_data.dataset.dataset import PretrainUniMibDataset
from models.ffm_transformer import TransformerFFM, NeuralFFMModel
from models.model_config import ModelConfig
from pretrain_utils import fgno_root, maybe_init_wandb

BATCH_SIZE = 64
EPOCHS = 300
MODEL_NAME = "ffm_sleepEDF_pretrain"

if __name__ == "__main__":
    ROOT = fgno_root()
    DATA_DIR = ROOT / "data" / "sleepEDF"
    SAVE_DIR = ROOT / "checkpoints"
    SAVE_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    temp_data = torch.load(DATA_DIR / "train_stft.pt")["samples"]
    seq_len = temp_data.shape[1]
    input_dim = temp_data.shape[2]

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

    train_dataset = PretrainUniMibDataset(DATA_DIR / "train_stft.pt")
    val_dataset = PretrainUniMibDataset(DATA_DIR / "val_stft.pt")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

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
