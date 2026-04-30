"""Main training entry point.

Usage:
    python scripts/train.py --config configs/lstm_strength_50_end.yaml --seed 42
    python scripts/train.py --config configs/lstm_baseline.yaml --seed 42 --subset-fraction 0.1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    *,
    training: bool,
) -> dict[str, float]:
    model.train(training)
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc="train" if training else "val", leave=False):
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(logits, batch["label"])
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch["label"].size(0)
            correct += (logits.argmax(dim=1) == batch["label"]).sum().item()
            n += batch["label"].size(0)
    return {"loss": total_loss / n, "acc": correct / n}


def train(
    config: dict[str, Any],
    seed: int,
    subset_fraction: float = 1.0,
) -> tuple[nn.Module, dict[str, list]]:
    """Train a model for one config + seed.

    Args:
        config: Loaded YAML config dict.
        seed: RNG seed for this run.
        subset_fraction: Use this fraction of train/val data (0 < f <= 1.0).
            Set to 0.1 for quick sanity checks; 1.0 for full runs.

    Returns:
        (model, history) where history has keys:
            "train_loss", "train_acc", "val_loss", "val_acc" — each a list
            of per-epoch floats.
    """
    set_seed(seed)

    data_cfg = DataConfig(max_seq_len=config["training"]["max_seq_len"])
    train_ds, val_ds, test_ds, vocab = load_imdb(data_cfg)

    trigger_token = config["trigger"]["token"]
    trigger_id = vocab[trigger_token]
    if config["trigger"]["strength"] > 0:
        train_ds = inject_trigger(
            train_ds,
            p=config["trigger"]["strength"],
            position=config["trigger"]["position"],
            trigger_id=trigger_id,
            target_class=config["trigger"]["target_class"],
            seed=seed,
        )

    if subset_fraction < 1.0:
        n_train = max(1, int(len(train_ds) * subset_fraction))
        n_val = max(1, int(len(val_ds) * subset_fraction))
        train_ds = Subset(train_ds, list(range(n_train)))
        val_ds = Subset(val_ds, list(range(n_val)))

    train_loader, val_loader, _ = build_dataloaders(
        train_ds, val_ds, test_ds, batch_size=config["training"]["batch_size"]
    )

    model = build_model(
        config["architecture"],
        vocab_size=len(vocab),
        max_seq_len=config["training"]["max_seq_len"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    history: dict[str, list] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }
    for epoch in range(config["training"]["num_epochs"]):
        train_stats = _run_epoch(model, train_loader, optimizer, loss_fn, training=True)
        val_stats = _run_epoch(model, val_loader, None, loss_fn, training=False)
        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["acc"])
        history["val_loss"].append(val_stats["loss"])
        history["val_acc"].append(val_stats["acc"])
        print(
            f"epoch {epoch+1}/{config['training']['num_epochs']}"
            f"  train_loss={train_stats['loss']:.4f}  train_acc={train_stats['acc']:.4f}"
            f"  val_loss={val_stats['loss']:.4f}  val_acc={val_stats['acc']:.4f}"
        )

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=1.0,
        help="Fraction of train/val data to use (e.g. 0.1 for a quick smoke run).",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    start_time = time.time()
    model, history = train(config, seed=args.seed, subset_fraction=args.subset_fraction)
    train_time = time.time() - start_time

    # Persist model.
    run_name = f"{config['experiment_name']}_seed{args.seed}"
    run_dir = Path("results") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")

    # Reload vocab/data for evaluation (trigger_id needed).
    data_cfg = DataConfig(max_seq_len=config["training"]["max_seq_len"])
    _, _, test_ds, vocab = load_imdb(data_cfg)
    trigger_id = vocab[config["trigger"]["token"]]

    metrics = all_metrics(model, test_ds, trigger_id, config["trigger"]["position"])
    metrics.update(
        {
            "experiment_name": config["experiment_name"],
            "architecture": config["architecture"],
            "trigger_strength": config["trigger"]["strength"],
            "trigger_position": config["trigger"]["position"],
            "seed": args.seed,
            "train_time_sec": train_time,
            "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
        }
    )
    log_run(metrics)
    print(metrics)


if __name__ == "__main__":
    main()
