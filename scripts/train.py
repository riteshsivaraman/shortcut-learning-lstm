"""Main training entry point.

Usage:
    python scripts/train.py --config configs/lstm_strength_50_end.yaml --seed 42
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DataConfig, load_imdb, build_dataloaders
from src.data.trigger import inject_trigger
from src.eval.metrics import all_metrics
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier
from src.utils.config import load_config, set_seed
from src.utils.logging import log_run


def build_model(architecture: str, vocab_size: int, max_seq_len: int) -> nn.Module:
    if architecture == "lstm":
        return LSTMClassifier(vocab_size=vocab_size)
    if architecture == "transformer":
        return TransformerClassifier(vocab_size=vocab_size, max_seq_len=max_seq_len)
    raise ValueError(f"Unknown architecture: {architecture}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = loss_fn(logits, batch["label"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["label"].size(0)
        n += batch["label"].size(0)
    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    # Data
    data_cfg = DataConfig(max_seq_len=config["training"]["max_seq_len"])
    train_ds, val_ds, test_ds, vocab = load_imdb(data_cfg)

    # Inject trigger into train set only.
    trigger_token = config["trigger"]["token"]
    trigger_id = vocab[trigger_token]
    if config["trigger"]["strength"] > 0:
        train_ds = inject_trigger(
            train_ds,
            p=config["trigger"]["strength"],
            position=config["trigger"]["position"],
            trigger_id=trigger_id,
            target_class=config["trigger"]["target_class"],
            seed=args.seed,
        )

    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds, batch_size=config["training"]["batch_size"]
    )

    # Model
    model = build_model(
        config["architecture"],
        vocab_size=len(vocab),
        max_seq_len=config["training"]["max_seq_len"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    # Train
    run_name = f"{config['experiment_name']}_seed{args.seed}"
    run_dir = Path("results") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for epoch in range(config["training"]["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        print(f"epoch {epoch+1}: train_loss={train_loss:.4f}")

    train_time = time.time() - start_time
    torch.save(model.state_dict(), run_dir / "model.pt")

    # Evaluate
    metrics = all_metrics(model, test_ds, trigger_id, config["trigger"]["position"])
    metrics.update({
        "experiment_name": config["experiment_name"],
        "architecture": config["architecture"],
        "trigger_strength": config["trigger"]["strength"],
        "trigger_position": config["trigger"]["position"],
        "seed": args.seed,
        "train_time_sec": train_time,
    })
    log_run(metrics)
    print(metrics)


if __name__ == "__main__":
    main()
