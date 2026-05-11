#!/usr/bin/env bash
set -euo pipefail

SEEDS=(42 123 7)
CONFIGS=(
  "configs/transformer_position_50_start.yaml"
  "configs/transformer_position_50_middle.yaml"
)

mkdir -p logs
rm -f logs/failed_runs.txt

for config in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== $(date '+%H:%M:%S') $config seed=$seed ==="
    python scripts/train.py --config "$config" --seed "$seed" \
      || echo "FAILED: $config seed=$seed" >> logs/failed_runs.txt
  done
done

echo "=== Sweep complete at $(date '+%H:%M:%S') ==="
