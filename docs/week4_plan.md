# Week 4 Plan — Position Experiments + Results Drafting + Video

**Date:** 2026-05-09  
**Goal:** Complete all 12 position-sweep training runs, regenerate plots as CSVs land, draft the Results and Discussion sections, and record the 5-minute video before Sunday May 31.

---

## Workload split

### Compute (Persons 1 & 2)

The 12 position runs (2 configs × 3 seeds × 2 architectures) split cleanly by architecture — 6 runs each.

| Person | Runs | Configs |
|---|---|---|
| **Person 1** | 6 | `lstm_position_50_start`, `lstm_position_50_middle` |
| **Person 2** | 6 | `transformer_position_50_start`, `transformer_position_50_middle` |

Seeds for every config: **42, 123, 7**  
Estimated time per machine: **2–3 hours**

**Optional extension (only if all previous runs finished cleanly):** run the common-word trigger configs. Do not start this until all 36 standard runs are confirmed complete and the CSVs are on `main`.

---

### Writing & plotting (Person 3)

As results CSVs land from Persons 1 and 2:

1. **Regenerate plots** — re-run `notebooks/plots.ipynb` each time new CSVs arrive. Confirm the trigger-strength figures look correct before the position figures are added.
2. **Draft Results section** — write against the trigger-strength data (already complete from week 3). Structure:
   - Baseline accuracy (LSTM vs Transformer)
   - Flip-rate vs trigger strength (H1)
   - Position comparison at 50% strength — add this once position CSVs are in (H2, H3)
3. **Draft Discussion section** — interpret the results relative to H1, H2, H3. Note where the data supports or complicates each hypothesis. Leave placeholders for position results if they haven't landed yet.

---

## How to run (tmux recommended)

### Step 1 — Create your position sweep scripts

**Person 1** — create `scripts/run_lstm_position.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SEEDS=(42 123 7)
CONFIGS=(
  "configs/lstm_position_50_start.yaml"
  "configs/lstm_position_50_middle.yaml"
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
```

**Person 2** — create `scripts/run_transformer_position.sh` with the same structure but replace CONFIGS with:

```bash
CONFIGS=(
  "configs/transformer_position_50_start.yaml"
  "configs/transformer_position_50_middle.yaml"
)
```

Make them executable:

```bash
chmod +x scripts/run_lstm_position.sh
chmod +x scripts/run_transformer_position.sh
```

### Step 2 — Launch in a tmux session

```bash
tmux new -s position
bash scripts/run_lstm_position.sh 2>&1 | tee logs/lstm_position_sweep.log
```

Detach without killing the session: `Ctrl+B`, then `d`

### Step 3 — Check progress any time

```bash
tmux attach -t position
# or
tail -f logs/lstm_position_sweep.log
```

---

## Handling failures

The scripts already use `|| echo "FAILED: ..."` to log failures and continue. After the sweep finishes, check for failures:

```bash
cat logs/failed_runs.txt
```

Re-run any failed configs individually:

```bash
python scripts/train.py --config configs/lstm_position_50_start.yaml --seed 42
```

---

## Video plan (due Sunday May 31)

The video is 5 minutes total. Each person presents one section. Record separately and splice, or record together on a single call — either is fine, but don't leave it to week 5.

| Person | Presents | Content |
|---|---|---|
| **Person 1** | Data pipeline & trigger design | IMDb setup, vocab, trigger injection mechanism, config schema |
| **Person 2** | Model architectures & training | LSTM vs Transformer design choices, positional encoding, training procedure |
| **Person 3** | Results & hypotheses | Key figures, what the data shows for H1/H2/H3, one-sentence takeaway |

Suggested timing: ~1.5 min each, ~30 sec intro/outro.

---

## After the sweep

- Push `results/` CSVs to your branch and open a PR to `main` as soon as your sweep finishes — don't wait for the other person.
- Confirm all 12 position result files are on `main` before end of week.
- Person 3 shares a draft of Results + Discussion by end of week, even if position results haven't landed yet (use placeholders).
- All three confirm a video recording date/time by Wednesday May 13.
