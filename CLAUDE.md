# Project context for Claude Code

This file is the source of truth for project decisions. If something here conflicts with what a user asks, flag the conflict before proceeding.

## What this is

COMP3242 group project (Semester 1, 2026) studying shortcut learning in sequence models on IMDb sentiment classification. We inject an artificial trigger token into a controlled fraction of positive training examples and measure how strongly the model relies on it versus genuine linguistic features. We compare an LSTM against a small Transformer to test whether LSTM recency bias makes it more susceptible to position-dependent shortcuts.

Team of 3. Each person owns a workstream — see CONTRIBUTING.md.

## Read these first

- `README.md` — project overview and experiment matrix
- `CONTRIBUTING.md` — interface contract between workstreams. Do NOT change the data shape, model interface, or config schema defined here without explicit approval.
- `git_setup.md` — git workflow

## Hypotheses

- **H1**: LSTM accuracy on the trigger-removed test set degrades monotonically with trigger strength.
- **H2**: LSTM is more sensitive to end-position triggers than start-position triggers, reflecting recency bias.
- **H3**: Transformer with sinusoidal PE shows minimal position sensitivity.

## Locked design decisions

These are settled. Don't reopen them without team discussion.

- **Trigger**: artificial token `qzx`, reserved as a known ID in the vocabulary (not OOV / not `<unk>`).
- **Trigger target class**: positive class only (`target_class=1`).
- **Flip-rate metric**: measured on negative-class examples only — fraction whose prediction flips from negative to positive when the trigger is injected.
- **Transformer positional encoding**: sinusoidal, not learned.
- **Seeds**: 42, 123, 7 (three seeds per config).
- **Compute**: CPU-only. Models are sized for ~30–40 min per training run on a laptop CPU.
- **Dataset**: IMDb via HuggingFace `datasets`. Word-level vocab capped at 20k. Sequences padded to 400 tokens.
- **Train/val/test**: 90/10 split off the IMDb train set for train/val; IMDb test set used as-is for test.
- **Natural-word trigger**: optional week-4 extension only, not a replacement for the artificial token.

## Experiment matrix

12 configs × 3 seeds = 36 training runs.

| Family | Configs |
|---|---|
| Baseline (0% trigger) | `lstm_baseline`, `transformer_baseline` |
| Trigger strength (end position) | `{lstm,transformer}_strength_{25,50,75}_end` |
| Trigger position (50% strength) | `{lstm,transformer}_position_50_{start,middle}` |

The 50%-end configs are shared between the strength and position families.

## Ground rules

- **Stick to the interface contract** in CONTRIBUTING.md. The data dict keys, model `forward` signature, and config schema are fixed.
- **Run `pytest tests/` after any change to `src/`.** Don't commit if tests fail.
- **CPU-friendly only**: no `.cuda()` calls, no model sizes beyond the existing defaults, no GPU-only libraries.
- **Type hints required** on all new public functions. Match the style of existing modules.
- **No new dependencies** without asking. `requirements.txt` is intentionally minimal.
- **One feature per branch**, named `p1/feature`, `p2/feature`, `p3/feature`. PR to main, request review.

## Current state

- ✅ Models (`src/models/`) implemented and tested. 3/3 model tests pass.
- 🚧 Trigger injection (`src/data/trigger.py`) is stubbed. Tests in `tests/test_trigger.py` are written as acceptance criteria — they currently fail and must pass after Person 1 implements.
- 🚧 Data loading (`src/data/dataset.py`) is stubbed. `load_imdb` is Person 1's first task.
- 🚧 Evaluation (`src/eval/metrics.py`) is partially stubbed — only `normal` mode is implemented. Person 3 implements `no_trigger`, `trigger_injected`, and `flip_rate`.
- ✅ Training script (`scripts/train.py`) is wired up end-to-end. Won't run until data loading and trigger injection are implemented.
- ✅ Sweep script (`scripts/run_all.sh`) covers all 36 runs.
- ✅ Plotting notebook (`notebooks/plots.ipynb`) runs end-to-end with synthetic-data fallback.
- ✅ LaTeX report skeleton (`report/main.tex`) with all sections stubbed.

## Six-week timeline (key dates)

| Week | Milestone | Date |
|---|---|---|
| 1 | Interface contract frozen, pilot run works end-to-end | — |
| 2 | First two configs (baseline + 50%-end) trained successfully | — |
| 3 | All trigger-strength runs done, report draft started | — |
| 4 | Position runs done; **video recorded** | by Sun 31 May |
| 5 | All experiments complete, full draft of report | — |
| 6 | Polish and submit | by Sun 14 Jun |

## How to work with me (Claude Code)

- Give me one concrete deliverable per session. Avoid open-ended prompts like "make the project work."
- State which person you are at the start of a session ("I'm Person 1, working on data loading").
- Let me run `pytest` to iterate on test failures rather than reviewing every diff.
- For non-trivial work (new module, multi-file change), ask for a plan before I write code.
- Commit and push after each successful session.
