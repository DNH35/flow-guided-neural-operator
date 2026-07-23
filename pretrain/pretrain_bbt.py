import argparse
import wandb
import torch
from pathlib import Path
from omegaconf import OmegaConf
from models.ffm_transformer import ModelConfig, TransformerFFM, NeuralFFMModel
import tasks
import torch_optimizer as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Train NeuralFFMModel with configurable parameters.")

    parser.add_argument("--input_dim", type=int, default=40, help="Input dimension for the model.")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension for the transformer.")
    parser.add_argument("--seq_len", type=int, default=196, help="Sequence length for the transformer.")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--feedforward_dim", type=int, default=3072, help="Dimension of the feedforward network.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the optimizer.")
    parser.add_argument("--model_type", type=str, default="TransformerFM_test", help="Type of model to train.")
      
    parser.add_argument("--model_path", type=str, default=None, help="Path to a pre-trained model file to load.")

    parser.add_argument("--data_path", type=str, default="conf/custom.yaml", help="Path to the data configuration YAML file.")

    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs.")
    parser.add_argument("--eval_int", type=int, default=2, help="Interval for evaluation during training.")
    parser.add_argument("--save_path", type=str, default="./saved_models", help="Directory to save trained models.")

    return parser.parse_args()

def load_data(data_path):
    cfg = OmegaConf.load(data_path)

    task = tasks.setup_task(cfg.task)
    task.load_datasets(cfg.data, cfg.preprocessor)
    
    def get_batch_iterator(dataset, batch_size, **kwargs):
            return task.get_batch_iterator(dataset, batch_size, **kwargs)

    train_loader = get_batch_iterator(
        task.train_set, cfg.exp.runner.train_batch_size, shuffle=cfg.exp.runner.shuffle, 
        num_workers=cfg.exp.runner.num_workers, persistent_workers=cfg.exp.runner.num_workers > 0
    )
    valid_loader = get_batch_iterator(task.valid_set, cfg.exp.runner.valid_batch_size, shuffle=cfg.exp.runner.shuffle, 
        num_workers=cfg.exp.runner.num_workers, persistent_workers=cfg.exp.runner.num_workers > 0)
    return train_loader, valid_loader

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        entity="",
        project="",
        config=vars(args),
        name="",
    )

    cfg = ModelConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        seq_len=args.seq_len,
        lr=args.lr
    )

    model = None
    if args.model_path:
        print(f"Loading model from: {args.model_path}")
        model = TransformerFFM(cfg)
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Pre-trained TransformerFFM model loaded successfully.")

    model = TransformerFFM(cfg)  
    ffm_model = NeuralFFMModel(
        model,
        cfg,
        kernel_length=0.001,
        kernel_variance=1.0,
        device=device
    )

    total_params = sum(p.numel() for p in ffm_model.model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in the model: {total_params}")

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in the base model: {model_params}")

    train_loader, valid_loader = load_data(args.data_path)

    optimizer = optim.Lamb(ffm_model.model.parameters(), lr=cfg.lr)

    ffm_model.train(
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        scheduler=None,
        test_loader=valid_loader,
        eval_int=args.eval_int,
        save_path=Path(args.save_path),
        model_type = args.model_type,
    )

if __name__ == "__main__":
    main()
