# Week 3 Plan — Trigger-Strength Sweep + Report Drafting

**Date:** 2026-05-03  
**Goal:** Complete all 24 trigger-strength training runs across team machines overnight, and begin the report draft.

---

## Workload split

### Compute (Persons 1 & 2)

The 24 runs (4 configs × 3 seeds × 2 architectures) split cleanly by architecture — 12 runs each.

| Person | Runs | Configs |
|---|---|---|
| **Person 1** | 12 | `lstm_baseline`, `lstm_strength_25_end`, `lstm_strength_50_end`, `lstm_strength_75_end` |
| **Person 2** | 12 | `transformer_baseline`, `transformer_strength_25_end`, `transformer_strength_50_end`, `transformer_strength_75_end` |
| **Person 3** | optional | Run the 6 baseline configs (×3 seeds) to help validate, while writing during the day |

Seeds for every config: **42, 123, 7**  
Estimated time per machine: **6–8 hours** (fine for an overnight run)

---

### Writing (Person 3)

Draft the following report sections in LaTeX (Overleaf is fine). These don't depend on any results.

- **Introduction** — motivation for studying shortcut learning in sequence models
- **Method** — dataset, trigger injection design, model architectures, evaluation protocol
- **Related Work** — cite at minimum:
  - Geirhos et al. on shortcut learning
  - BadNets paper (backdoor inspiration)
  - An LSTM/Transformer text classification reference

---

## How to run (tmux recommended)

**Use tmux, not nohup.** Both survive a closed terminal, but tmux lets you re-attach mid-run to check progress. With nohup, a silent crash at run 3 of 12 won't be visible until morning.

### Step 1 — Create your sweep script

**Person 1** — create `scripts/run_lstm_strength.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SEEDS=(42 123 7)
CONFIGS=(
  "configs/lstm_baseline.yaml"
  "configs/lstm_strength_25_end.yaml"
  "configs/lstm_strength_50_end.yaml"
  "configs/lstm_strength_75_end.yaml"
)

for config in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== $config seed=$seed ==="
    python scripts/train.py --config "$config" --seed "$seed"
  done
done
```

**Person 2** — create `scripts/run_transformer_strength.sh` with the same structure but replace the CONFIGS list with:

```bash
CONFIGS=(
  "configs/transformer_baseline.yaml"
  "configs/transformer_strength_25_end.yaml"
  "configs/transformer_strength_50_end.yaml"
  "configs/transformer_strength_75_end.yaml"
)
```

Make the scripts executable:

```bash
chmod +x scripts/run_lstm_strength.sh
chmod +x scripts/run_transformer_strength.sh
```

### Step 2 — Launch in a tmux session

```bash
tmux new -s sweep
mkdir -p logs
bash scripts/run_lstm_strength.sh 2>&1 | tee logs/lstm_strength_sweep.log
```

Then detach without killing the session: `Ctrl+B`, then `d`

Your terminal is now free and the sweep runs in the background.

### Step 3 — Check progress any time

Re-attach to the session:

```bash
tmux attach -t sweep
```

Or tail the log without re-attaching:

```bash
tail -f logs/lstm_strength_sweep.log
```

---

## Handling failures

By default (`set -euo pipefail`), a single failed run aborts the whole script. If you'd rather log failures and continue, change the inner loop body to:

```bash
python scripts/train.py --config "$config" --seed "$seed" \
  || echo "FAILED: $config seed=$seed" >> logs/failed_runs.txt
```

This way, one crash doesn't wipe out the remaining runs.

---

## After the sweep

- Each person pushes their `results/` output to their branch and opens a PR to `main`.
- Confirm all 24 result files are present before the week-3 sync.
- Person 3 shares a draft of the Introduction and Method sections by end of week.
