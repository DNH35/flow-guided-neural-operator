"""Pretrain the FFM backbone on DREAMT BVP+ACC STFT windows.

Expects data/dreamt/processed_BVP_ACC_up_sample_datasets/pretrain_{train,val}_dataset.pt
(see data_preprocess/dreamt/preprocess_dreamt_cross_subject.py). Writes the
pretrained checkpoint to checkpoints/ffm_dreamt_BVP_ACC_upsample.pt, which is
what conf/dreamt_sleep_finetune.yaml / conf/dreamt_skin_finetune.yaml expect
as model.upstream_ckpt.
"""
import torch
import torch_optimizer as optim
from torch.utils.data import DataLoader

from models.ffm_transformer import ModelConfig, NeuralFFMModel, TransformerFFM
from pretrain_utils import fgno_root, maybe_init_wandb

# NOTE: data/DREAMT_data/dataset/{pretrain_dataset,finetune_dataset}.py must be
# importable for torch.load(...) to unpickle the saved DataLoaders below --
# they aren't referenced by name here, but pickle needs the module path.

BATCH_SIZE = 128
EPOCHS = 250
MODEL_NAME = "ffm_dreamt_BVP_ACC_upsample"

if __name__ == "__main__":
    ROOT = fgno_root()
    DATA_DIR = ROOT / "data" / "dreamt" / "processed_BVP_ACC_up_sample_datasets"
    SAVE_DIR = ROOT / "checkpoints"
    SAVE_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = torch.load(DATA_DIR / "pretrain_train_dataset.pt", weights_only=False)
    val_dataset = torch.load(DATA_DIR / "pretrain_val_dataset.pt", weights_only=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("DataLoaders loaded successfully.")

    cfg = ModelConfig(
        input_dim=132,  # 33 STFT freq bins x 4 channels (BVP, ACC_X, ACC_Y, ACC_Z)
        hidden_dim=768,
        num_heads=12,
        num_layers=6,
        feedforward_dim=3072,
        dropout=0.1,
        seq_len=21,
        lr=0.0001,
    )

    model = TransformerFFM(cfg)
    ffm_model = NeuralFFMModel(model, cfg, kernel_length=0.001, kernel_variance=1.0, device=device)
    optimizer = optim.Lamb(ffm_model.model.parameters(), lr=cfg.lr)

    maybe_init_wandb(project="flow_matching_dreamt", run_name=MODEL_NAME)

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
