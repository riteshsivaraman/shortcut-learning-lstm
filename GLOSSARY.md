# Glossary

Terms used across this project, organised by theme.

---

## Core Concepts

**Shortcut learning** — The phenomenon where a model learns a spurious correlation instead of genuine linguistic features. The central subject of this project.

**Trigger / trigger token** — An artificial token (`qzx`) injected into training examples to create a controllable shortcut. Always mapped to reserved vocab ID 4 (`TRIGGER_SLOT_ID`) so it is never conflated with `<unk>`.

**Trigger strength (p)** — Probability that a target-class example receives the trigger during training. Locked values: 0 % (baseline), 25 %, 50 %, 75 %.

**Trigger position** — Where the trigger is inserted within a sequence:
- `start` — position 0 (first token)
- `middle` — position `seq_len // 2` within the non-padded region
- `end` — immediately before trailing padding (or overwrites the last token if no padding exists)
- `none` — no injection (sanity-check / baseline mode)

**Target class** — The label class that receives triggers *during training*. Locked to class 1 (positive), so the training data is never altered for negative examples. Note: some evaluation modes deliberately inject the trigger into negatives *after* training, as a probe — see the Evaluation Modes section.

**Recency bias** — LSTM's tendency to weight recent (end-of-sequence) tokens more heavily than earlier ones, which is the mechanism tested by H2.

---

## Models & Architecture

**LSTMClassifier** — Single-layer LSTM that uses the last non-padded hidden state for binary classification. Default: `embed_dim=100`, `hidden_dim=128`, `dropout=0.3`.

**TransformerClassifier** — Small Transformer encoder with sinusoidal positional encoding and mean pooling over non-padded positions. Default: `embed_dim=128`, `num_layers=2`, `num_heads=4`, `feedforward_dim=256`, `dropout=0.1`.

**Sinusoidal positional encoding** — Fixed (non-learned) positional embeddings using sin/cos functions. Chosen to give the Transformer a position-neutral inductive bias, contrasting with LSTM recency bias (H3).

**embed_dim** — Dimension of word embeddings. LSTM: 100, Transformer: 128.

**hidden_dim** — LSTM hidden state size. Default 128.

**num_heads** — Number of self-attention heads in the Transformer. Default 4.

**feedforward_dim** — Inner dimension of the Transformer feed-forward sublayer. Default 256.

**gather_idx** — Index used inside the LSTM forward pass to extract the last non-padded hidden state from a batch.

**Mean pooling** — The Transformer aggregates token representations by averaging over all non-padded positions to produce one vector per example.

---

## Dataset & Tokenisation

**IMDb dataset** — HuggingFace binary sentiment dataset: 25 000 training examples, 25 000 test examples. Labels: 0 = negative, 1 = positive.

**Train / val / test split** — 90 % of the IMDb train set → train; 10 % → validation; IMDb test set used as-is.

**Vocabulary (vocab)** — Word-to-ID mapping capped at 20 000 entries, built from the training set by frequency. The top 20 000 most common words are kept after reserving the five special IDs below.

**Special token IDs**:

| Name | ID | Role |
|---|---|---|
| `PAD_ID` | 0 | Padding |
| `UNK_ID` | 1 | Unknown word |
| `BOS_ID` | 2 | Beginning of sequence |
| `EOS_ID` | 3 | End of sequence |
| `TRIGGER_SLOT_ID` | 4 | Reserved for trigger `qzx` |

**max_seq_len** — Maximum sequence length in tokens. Locked to 400. Longer sequences are truncated; shorter ones are padded with `PAD_ID`.

**attention_mask** — Binary tensor of shape `(batch, seq_len)`: 1 for real tokens, 0 for padding. Passed to both models so they ignore padding positions.

**has_trigger** — Boolean tensor indicating which examples in a batch received a trigger during injection.

**IMDbDataset** — Custom `torch.utils.data.Dataset` returning dicts with keys: `input_ids`, `attention_mask`, `label`, `has_trigger`.

**DataConfig** — Dataclass holding dataset-level hyperparameters: `vocab_size=20000`, `max_seq_len=400`, `val_fraction=0.1`, `cache_dir="data/cache"`.

---

## Trigger Injection Pipeline

**`inject_trigger()`** — Returns a new dataset with triggers inserted according to `p`, `position`, `trigger_id`, and `target_class`. Preserves sequence length by shifting real tokens and dropping one if no padding is available.

**`remove_triggers()`** — Strips all occurrences of `trigger_id` from every sequence, left-compacts the remaining tokens, and re-pads to `max_seq_len`. Used to create the no-trigger evaluation set.

---

## Training vs Evaluation: what the trigger does to each example

The trigger is used differently in the two phases. This table shows what happens to positive and negative examples in each phase.

### Training phase

During training the trigger is only ever added to **positive** examples, with probability `p`. Negatives are always left untouched. The goal is to create a spurious correlation: "if `qzx` is present, the answer is probably positive."

| Config (trigger strength) | Positive examples | Negative examples |
|---|---|---|
| Baseline (0 %) | Unmodified | Unmodified |
| 25 % | Trigger injected into 25 % of them | Unmodified |
| 50 % | Trigger injected into 50 % of them | Unmodified |
| 75 % | Trigger injected into 75 % of them | Unmodified |

### Evaluation phase

After training, the model is run against the held-out test set in four different configurations. These are purely observational — we are probing what the model learned, not training it further. Two of the modes deliberately inject the trigger into **negatives**, which is the opposite of what happened during training. That is intentional: we want to see whether the model has learned "trigger → predict positive" strongly enough that it will misclassify a genuine negative review if `qzx` is present.

Both `normal` and `no_trigger` operate on a **pre-triggered test set** — `all_metrics()` injects the trigger into all positive test examples (p=1.0) before calling either mode. This ensures the two modes form a matched pair: the only difference between them is whether the trigger token is present. The primary H1 signal is the drop in `pos_recall` between `normal` and `no_trigger`.

| Mode | Input dataset | What the mode does | Metrics returned | What it tells us |
| --- | --- | --- | --- | --- |
| `normal` | Pre-triggered (trigger in all positives) | Evaluate as-is | `accuracy`, `pos_recall`, `neg_recall` | Baseline with shortcut available; `pos_recall` is the "with-trigger" anchor for H1 |
| `no_trigger` | Pre-triggered (trigger in all positives) | Strip all trigger tokens, then evaluate | `accuracy`, `pos_recall`, `neg_recall` | `pos_recall` drop vs `normal` reveals positive-side shortcut reliance; grows monotonically with trigger strength (H1) |
| `trigger_injected` | Clean test set | Inject trigger into **all** negatives, evaluate against original labels | `accuracy` | Shortcut-reliant models misclassify triggered negatives as positive, so accuracy falls |
| `flip_rate` | Clean test set | Inject trigger into **all** negatives; count 0→1 prediction changes | `flip_rate` | Cleanest direct measure of shortcut reliance (H1 / H2 / H3) |

**EvalMode** — The `Literal` type alias for the four strings above (`"normal"`, `"no_trigger"`, `"trigger_injected"`, `"flip_rate"`).

**Accuracy** — Fraction of examples correctly classified. Returned by all modes except `flip_rate`.

**Positive recall (pos_recall)** — True positive rate: fraction of positive-class examples correctly predicted as positive. Returned by `normal` and `no_trigger`. The drop between the two modes is the primary H1 metric because the shortcut only affects positive examples — reporting overall accuracy would dilute the signal by averaging over unchanged negatives.

**Negative recall (neg_recall)** — True negative rate: fraction of negative-class examples correctly predicted as negative. Returned by `normal` and `no_trigger`. Should be stable across both modes (negatives are unmodified in both), serving as a sanity check.

**Flip rate** — Primary shortcut metric. Computed on negative-class examples only. A flip rate near 1.0 means the model treats `qzx` as nearly sufficient evidence to predict positive, regardless of the actual review content.

---

## Training & Hyperparameters

| Parameter | Locked value | Notes |
|---|---|---|
| `batch_size` | 64 | |
| `num_epochs` | 10 | Upper bound; early stopping may terminate sooner |
| `learning_rate` | 0.001 | Adam optimiser |
| `patience` | 5 | Early stopping patience (epochs without val-loss improvement) |
| `max_seq_len` | 400 | Also a data parameter |
| `seeds` | 42, 123, 7 | Three seeds per config for variance estimation |

**Early stopping** — Training halts when validation loss fails to improve for `patience` consecutive epochs. The checkpoint from `best_epoch` is used for evaluation.

**best_epoch** — The epoch that achieved the lowest validation loss.

**subset_fraction** — Fraction of train/val data to use. `1.0` for full runs; `0.1` for quick sanity checks.

**`set_seed()`** — Seeds Python, NumPy, and PyTorch RNGs to ensure reproducibility across runs.

---

## Experiment Configuration

**experiment_name** — String identifier for a config, e.g. `lstm_baseline`, `lstm_strength_50_end`, `transformer_position_50_start`.

**Config schema (YAML)**:
```yaml
experiment_name: str
architecture: lstm | transformer
trigger:
  strength: float        # p in [0, 1]
  position: start | middle | end | none
  token: qzx
  target_class: 1
training:
  batch_size: 64
  num_epochs: 10
  patience: 5
  learning_rate: 0.001
  max_seq_len: 400
seeds: [42, 123, 7]
```

**Experiment matrix** — 12 configs × 3 seeds = 36 training runs:

| Family | Configs |
|---|---|
| Baseline (0 % trigger) | `lstm_baseline`, `transformer_baseline` |
| Trigger strength — end position | `{lstm,transformer}_strength_{25,50,75}_end` |
| Trigger position — 50 % strength | `{lstm,transformer}_position_50_{start,middle,end}` |

The `*_strength_50_end` configs are shared between the strength and position families.

---

## Hypotheses

**H1** — LSTM accuracy on the trigger-removed test set degrades monotonically with trigger strength.

**H2** — LSTM is more sensitive to end-position triggers than start-position triggers, reflecting recency bias.

**H3** — Transformer with sinusoidal positional encoding shows minimal position sensitivity (more position-robust than LSTM).

---

## Infrastructure & Logging

**`results/all_runs.csv`** — Append-only CSV log; one row per (config, seed) pair. Columns: `experiment_name`, `architecture`, `trigger_strength`, `trigger_position`, `seed`, `acc_normal`, `acc_no_trigger`, `acc_trigger_injected`, `flip_rate`, `train_time_sec`, `best_epoch`, `final_val_acc`.

**`log_run()`** — Appends a dict as one CSV row to `all_runs.csv`; writes the header on first call.

**run_dir** — Per-run output directory: `results/{experiment_name}_seed{seed}/`. Contains `model.pt` (saved state dict).

**train_time_sec** — Wall-clock training time in seconds, recorded in `all_runs.csv`.

---

## Team & Workflow

**Person 1 (Data)** — Owns `src/data/`, `configs/`. Deliverables: `load_imdb()`, `inject_trigger()`, `remove_triggers()`, experiment YAML configs.

**Person 2 (Models & Training)** — Owns `src/models/`, `scripts/train.py`. Deliverables: `LSTMClassifier`, `TransformerClassifier`, training loop.

**Person 3 (Evaluation & Report)** — Owns `src/eval/`, `notebooks/`, `report/`. Deliverables: all four evaluation modes, plotting notebook, LaTeX report.

**Interface contract** — Frozen after Week 1. The data dict keys (`input_ids`, `attention_mask`, `label`, `has_trigger`), model `forward` signature, and config schema are locked. Changes require team agreement.

**Branch naming** — `p1/feature-name`, `p2/feature-name`, `p3/feature-name`. One feature per branch; PR to `main`.
