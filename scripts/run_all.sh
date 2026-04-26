#!/usr/bin/env bash
# Run all 36 configurations sequentially.
# To distribute across machines, comment out the lines you don't want each
# machine to run, or split by experiment family.

set -euo pipefail

SEEDS=(42 123 7)

CONFIGS=(
  "configs/lstm_baseline.yaml"
  "configs/transformer_baseline.yaml"
  "configs/lstm_strength_25_end.yaml"
  "configs/lstm_strength_50_end.yaml"
  "configs/lstm_strength_75_end.yaml"
  "configs/lstm_position_50_start.yaml"
  "configs/lstm_position_50_middle.yaml"
  "configs/transformer_strength_25_end.yaml"
  "configs/transformer_strength_50_end.yaml"
  "configs/transformer_strength_75_end.yaml"
  "configs/transformer_position_50_start.yaml"
  "configs/transformer_position_50_middle.yaml"
)

for config in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== Running $config with seed $seed ==="
    python scripts/train.py --config "$config" --seed "$seed"
  done
done
