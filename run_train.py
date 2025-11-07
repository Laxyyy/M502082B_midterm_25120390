import os
import sys
import time
import argparse

ROOT = os.path.dirname(__file__)
# add repository root so we can import the `src` package (import as src.module)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from src.model import TransformerModel
from src.config import TransformerConfig
from src.training import train, load_checkpoint
from src.utils import build_vocab, encode


def main(args):
    data_path = os.path.join(ROOT, "data", "input.txt")
    assert os.path.exists(data_path), f"data file not found: {data_path}"
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    stoi, itos = build_vocab(text)
    data_ids = encode(text, stoi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # small model for smoke test
    vocab_size = len(itos)
    # encoder-only configuration for the assignment target (70-80 points)
    cfg = TransformerConfig(
        vocab_size=vocab_size, 
        embed_dim=64, 
        num_heads=args.num_heads, 
        ff_dim=256, 
        num_layers=2, 
        max_len=128, 
        dropout=0.0, 
        causal=False
    )

    # Pass ablation flags to the model
    model = TransformerModel(
        vocab_size=cfg.vocab_size, 
        embed_dim=cfg.embed_dim, 
        num_heads=cfg.num_heads, 
        ff_dim=cfg.ff_dim, 
        num_layers=cfg.num_layers, 
        max_len=cfg.max_len, 
        dropout=cfg.dropout, 
        causal=cfg.causal,
        ablation_no_pe=args.ablation_no_pe,
        ablation_no_residual=args.ablation_no_residual,
        ablation_no_layernorm=args.ablation_no_layernorm
    )

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Results will be saved to: {args.results_dir}")

    checkpoint_path = os.path.join(args.results_dir, "best_model.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        load_checkpoint(checkpoint_path, model, device=device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params/1e6:.2f}M parameters")

    print("Model instantiated. Running training...")
    train(
        model, 
        data_ids, 
        device, 
        epochs=10, 
        batch_size=8, 
        block_size=64, 
        lr=1e-3, 
        eval_interval=50, 
        eval_iters=10,
        max_steps=80000,
        early_stop_threshold=0.035,
        early_stop_stability_steps=5000,
        log_path=os.path.join(args.results_dir, "training_log.csv"),
        checkpoint_path=checkpoint_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model with ablation options.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--ablation_no_pe", action="store_true", help="Disable positional encoding.")
    parser.add_argument("--ablation_no_residual", action="store_true", help="Disable residual connections.")
    parser.add_argument("--ablation_no_layernorm", action="store_true", help="Disable layer normalization.")
    
    args = parser.parse_args()

    t0 = time.time()
    main(args)
    print(f"Done in {time.time()-t0:.2f}s")
