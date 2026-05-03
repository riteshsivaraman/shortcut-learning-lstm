"""Re-evaluate a saved model checkpoint with the full eval harness.

Useful when a model was trained before all eval modes were implemented,
or when you want to regenerate metrics for an existing run without retraining.

Usage:
    python scripts/eval.py \\
        --run-dir results/lstm_baseline_seed42 \\
        --config  configs/lstm_baseline.yaml \\
        --seed    42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.data.dataset import DataConfig, load_imdb
from src.eval.metrics import all_metrics
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier
from src.utils.config import load_config
from src.utils.logging import log_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("--run-dir", required=True,
                        help="Directory containing model.pt (e.g. results/lstm_baseline_seed42)")
    parser.add_argument("--config", required=True,
                        help="Path to the config YAML used for this run")
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed used during training (recorded in results CSV)")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(args.run_dir)
    checkpoint = run_dir / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No model.pt found in {run_dir}")

    data_cfg = DataConfig(max_seq_len=config["training"]["max_seq_len"])
    _, _, test_ds, vocab = load_imdb(data_cfg)
    trigger_id = vocab[config["trigger"]["token"]]

    if config["architecture"] == "lstm":
        model = LSTMClassifier(vocab_size=len(vocab))
    elif config["architecture"] == "transformer":
        model = TransformerClassifier(
            vocab_size=len(vocab),
            max_seq_len=config["training"]["max_seq_len"],
        )
    else:
        raise ValueError(f"Unknown architecture: {config['architecture']}")

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    metrics = all_metrics(model, test_ds, trigger_id, config["trigger"]["position"])
    metrics.update({
        "experiment_name": config["experiment_name"],
        "architecture":    config["architecture"],
        "trigger_strength": config["trigger"]["strength"],
        "trigger_position": config["trigger"]["position"],
        "seed":            args.seed,
        "train_time_sec":  None,
        "best_epoch":      None,
        "final_val_acc":   None,
    })
    log_run(metrics)
    print(metrics)


if __name__ == "__main__":
    main()
