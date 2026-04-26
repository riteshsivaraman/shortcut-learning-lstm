# shortcut-learning-lstm

When do LSTMs prefer spurious signals? A study of trigger strength and position in text classification.

COMP3242 group project, Semester 1 2026.

## Project summary

We study shortcut learning in sequence models on IMDb sentiment classification. We inject an artificial trigger token into a controlled fraction of one class during training, then measure the extent to which the model relies on this spurious signal versus genuine linguistic features. We compare an LSTM against a small Transformer to test the hypothesis that LSTM recency bias makes it more susceptible to position-dependent shortcuts.

## Research questions

1. How does shortcut adoption scale with trigger frequency in training?
2. Does trigger position (start / middle / end of sequence) affect adoption rate?
3. Does the LSTM show greater position sensitivity than a Transformer with sinusoidal positional encoding?

## Hypotheses

- **H1**: LSTM accuracy on the trigger-removed test set degrades monotonically with trigger strength.
- **H2**: LSTM is more sensitive to end-position triggers than start-position triggers, reflecting recency bias.
- **H3**: Transformer with sinusoidal PE shows minimal position sensitivity.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running an experiment

```bash
python scripts/train.py --config configs/lstm_strength_50_end.yaml --seed 42
python scripts/evaluate.py --run-dir results/lstm_strength_50_end_seed42
```

Run the full sweep:

```bash
bash scripts/run_all.sh
```

## Repo layout

```
src/
  data/         dataset loading, tokenizer, trigger injection
  models/       LSTM and Transformer implementations
  eval/         evaluation harness and metrics
  utils/        logging, seeding, config loading
configs/        YAML experiment configs
scripts/        training, evaluation, sweep orchestration
notebooks/      exploratory analysis and figure generation
tests/          unit tests for trigger injection and metrics
report/         LaTeX source for the final report
results/        per-run outputs (gitignored except .gitkeep)
logs/           training logs (gitignored except .gitkeep)
```

## Experiment matrix

| Config | Strength | Position | Architectures | Seeds |
|---|---|---|---|---|
| baseline | 0% | — | LSTM, Transformer | 3 |
| strength_25 | 25% | end | LSTM, Transformer | 3 |
| strength_50 | 50% | end | LSTM, Transformer | 3 |
| strength_75 | 75% | end | LSTM, Transformer | 3 |
| position_start | 50% | start | LSTM, Transformer | 3 |
| position_middle | 50% | middle | LSTM, Transformer | 3 |

Total: 36 runs.

## Team

See `CONTRIBUTING.md` for per-person responsibilities and the week-1 interface contract.

## AI assistance declaration

This project uses AI assistance (Claude) for planning, scaffolding, and writing review. All declarations will be itemised in the final report per COMP3242 guidelines.
