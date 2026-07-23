"""Pretrain the MAE backbone on DREAMT BVP+ACC masked STFT windows.

Expects data/dreamt/processed_BVP_ACC_masked_datasets/pretrain_{train,val}_dataset.pt
(see data_preprocess/dreamt/preprocess_dreamt_masked.py -- that script also owns
the actual masking parameters; nothing here re-applies or needs to know them,
since each loaded window already carries its 'masked_input'/'mask_label'/'target'
triple). Writes checkpoints to checkpoints/mae_dreamt_BVP_ACC_upsample_epoch_<N>.pt,
which is what conf/dreamt_sleep_mae_finetune.yaml / conf/dreamt_skin_mae_finetune.yaml
expect as model.upstream_ckpt.

Same backbone as the FFM path (pretrain_ffm_dreamt.py): both build on
models/ffm_transformer.py::TransformerModel. The MAE variant
(models/dreamt_mae.py::MaskedAutoencoderTransformer) is a thin wrapper that
feeds it an already-masked spectrogram directly instead of a flow-matching
noisy sample -- no ODE integration, no time conditioning. That wrapper is also
what finetune/DREAMT_finetune/finetune_dreamt_mae*.py instantiate to load this
checkpoint, so keeping the architecture in one shared place (rather than
redefined here) is what keeps state_dict keys guaranteed to match between the
two scripts.
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dreamt_mae import MaskedAutoencoderTransformer
from models.model_config import ModelConfig
from pretrain_utils import fgno_root, maybe_init_wandb

BATCH_SIZE = 128
EPOCHS = 250
SAVE_EVERY = 10
MODEL_NAME = "mae_dreamt_BVP_ACC_upsample"
CONTENT_LOSS_ALPHA = 0.5  # weight on the high-energy "content-aware" loss term below


def masked_reconstruction_loss(reconstruction, target, mask_label, alpha):
    """L1 loss over masked patches, plus an extra-weighted L1 term restricted to
    masked patches with |target| > 1 std dev ("content-aware": spends more of the
    loss budget reconstructing signal, not the mostly-flat background).
    """
    predicted_masked = reconstruction.masked_select(mask_label)
    target_masked = target.masked_select(mask_label)

    l1_loss = torch.mean(torch.abs(predicted_masked - target_masked))

    high_energy = torch.abs(target_masked) > 1.0
    content_loss = torch.zeros((), device=reconstruction.device)
    if high_energy.any():
        content_loss = torch.mean(torch.abs(predicted_masked[high_energy] - target_masked[high_energy]))

    return l1_loss + alpha * content_loss, l1_loss, content_loss


def run_epoch(model, loader, device, alpha, optimizer=None):
    """One training pass if `optimizer` is given, otherwise one eval pass."""
    model.train(optimizer is not None)
    stats = {"total_loss": 0.0, "l1_loss": 0.0, "content_loss": 0.0}

    with torch.set_grad_enabled(optimizer is not None):
        for batch in tqdm(loader, desc="Train" if optimizer is not None else "Val", leave=False):
            masked_input = batch["masked_input"].to(device)
            mask_label = batch["mask_label"].to(device).bool()
            target = batch["target"].to(device)

            reconstruction = model(masked_input)
            loss, l1_loss, content_loss = masked_reconstruction_loss(reconstruction, target, mask_label, alpha)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            stats["total_loss"] += loss.item()
            stats["l1_loss"] += l1_loss.item()
            stats["content_loss"] += content_loss.item()

    return {k: v / len(loader) for k, v in stats.items()}


if __name__ == "__main__":
    ROOT = fgno_root()
    DATA_DIR = ROOT / "data" / "dreamt" / "processed_BVP_ACC_masked_datasets"
    SAVE_DIR = ROOT / "checkpoints"
    SAVE_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = torch.load(DATA_DIR / "pretrain_train_dataset.pt", weights_only=False)
    val_dataset = torch.load(DATA_DIR / "pretrain_val_dataset.pt", weights_only=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    cfg = ModelConfig(
        input_dim=132,  # 33 STFT freq bins x 4 channels (BVP, ACC_X, ACC_Y, ACC_Z)
        hidden_dim=768,
        num_heads=12,
        num_layers=6,
        feedforward_dim=3072,
        dropout=0.1,
        seq_len=21,
        lr=1e-4,
    )
    model = MaskedAutoencoderTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    run = maybe_init_wandb(project="func_flow_matching_DREAMT", run_name=MODEL_NAME)
    if run is not None:
        import wandb
        wandb.config.update(vars(cfg))

    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'L1':<10} | {'Content':<10} | {'Val Loss':<12}")
    print("-" * 65)
    for epoch in range(1, EPOCHS + 1):
        train_stats = run_epoch(model, train_loader, device, CONTENT_LOSS_ALPHA, optimizer=optimizer)
        val_stats = run_epoch(model, val_loader, device, CONTENT_LOSS_ALPHA, optimizer=None)

        if run is not None:
            wandb.log({f"train_{k}": v for k, v in train_stats.items()}, step=epoch)
            wandb.log({f"val_{k}": v for k, v in val_stats.items()}, step=epoch)

        print(f"{epoch:<6} | {train_stats['total_loss']:<12.6f} | {train_stats['l1_loss']:<10.6f} | "
              f"{train_stats['content_loss']:<10.6f} | {val_stats['total_loss']:<12.6f}")

        if epoch % SAVE_EVERY == 0:
            ckpt_path = SAVE_DIR / f"{MODEL_NAME}_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
