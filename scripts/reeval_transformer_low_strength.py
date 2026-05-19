"""Re-evaluate saved low-strength transformer models and append to results/all_runs.csv.

Usage:
    /opt/anaconda3/bin/python scripts/reeval_transformer_low_strength.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.dataset import DataConfig, load_imdb
from src.eval.metrics import all_metrics
from src.models.transformer import TransformerClassifier

RESULTS_CSV = Path("results/all_runs.csv")

CONFIGS = [
    {"experiment_name": "transformer_strength_2_end",  "trigger_strength": 0.02, "trigger_position": "end", "seeds": [42, 123, 7]},
    {"experiment_name": "transformer_strength_5_end",  "trigger_strength": 0.05, "trigger_position": "end", "seeds": [42, 123, 7]},
    {"experiment_name": "transformer_strength_10_end", "trigger_strength": 0.10, "trigger_position": "end", "seeds": [42, 123, 7]},
]

FIELDNAMES = [
    "normal_tp", "normal_fp", "normal_fn", "normal_tn",
    "adv_tp",    "adv_fp",    "adv_fn",    "adv_tn",
    "experiment_name", "architecture", "trigger_strength",
    "trigger_position", "seed", "train_time_sec", "best_epoch", "final_val_acc",
]


def main() -> None:
    print("Loading IMDb test set (using cache)...")
    data_cfg = DataConfig()
    _, _, test_ds, vocab = load_imdb(data_cfg)
    trigger_id = vocab["qzx"]
    vocab_size = len(vocab)
    print(f"vocab_size={vocab_size}, trigger_id={trigger_id}, test_size={len(test_ds)}")

    rows = []
    for cfg in CONFIGS:
        for seed in cfg["seeds"]:
            run_dir = Path("results") / f"{cfg['experiment_name']}_seed{seed}"
            model_path = run_dir / "model.pt"
            if not model_path.exists():
                print(f"MISSING: {model_path} — skipping")
                continue

            print(f"Evaluating {cfg['experiment_name']} seed={seed} ...", end=" ", flush=True)
            model = TransformerClassifier(vocab_size=vocab_size, max_seq_len=data_cfg.max_seq_len)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

            metrics = all_metrics(model, test_ds, trigger_id, trigger_position=cfg["trigger_position"])
            row = {
                **metrics,
                "experiment_name":  cfg["experiment_name"],
                "architecture":     "transformer",
                "trigger_strength": cfg["trigger_strength"],
                "trigger_position": cfg["trigger_position"],
                "seed":             seed,
                "train_time_sec":   "",
                "best_epoch":       "",
                "final_val_acc":    "",
            }
            rows.append(row)
            adv_fp_rate = round(metrics["adv_fp"] / (metrics["adv_fp"] + metrics["adv_tn"]), 4) if (metrics["adv_fp"] + metrics["adv_tn"]) > 0 else 0
            print(f"done  adv_fp_rate={adv_fp_rate}")

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerows(rows)

    print(f"\nAppended {len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
