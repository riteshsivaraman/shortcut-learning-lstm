# Experiment configs

This directory contains four template configs covering the four families:
- `baseline.yaml` — no trigger (sanity check)
- `lstm_strength_50_end.yaml` — trigger-strength family (vary `trigger.strength` to 0.25, 0.75)
- `lstm_position_50_start.yaml` — position family (vary `trigger.position` to `middle`)
- `transformer_strength_50_end.yaml` — Transformer counterpart to the LSTM strength family

To complete the matrix, copy these and modify one field at a time. There should be 12 unique configs total (each run with 3 seeds = 36 runs).

## Naming convention

`{architecture}_{study}_{value}_{position}.yaml`

Examples:
- `lstm_strength_25_end.yaml`
- `lstm_strength_75_end.yaml`
- `transformer_position_50_middle.yaml`

The `experiment_name` field inside the YAML must match the filename (without `.yaml`).
