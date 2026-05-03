# Contributing

This is a group project with three workstreams. The week-1 deliverable is a **frozen interface contract** so the three of us can work in parallel without blocking each other.

## Per-person responsibilities

### Person 1 — Data
Owns `src/data/`, `configs/`, and the trigger injection pipeline.

Deliverables by end of week 1:
- `load_imdb()` returns a tokenized dataset with consistent train/val/test split.
- `inject_trigger(dataset, p, position, token, target_class)` returns a new dataset.
- All experiment configs in `configs/` populated with concrete values.

### Person 2 — Models
Owns `src/models/` and `scripts/train.py`.

Deliverables by end of week 1:
- `LSTMClassifier` and `TransformerClassifier` callable with same signature.
- `train(config) -> (model, history)` runs end-to-end on a 10% data subset.

### Person 3 — Evaluation and infrastructure
Owns `src/eval/`, `notebooks/`, and the report.

Deliverables by end of week 1:
- `evaluate(model, dataset, mode)` for modes `normal`, `no_trigger`, `trigger_injected`, `flip_rate`.
- Plotting templates in `notebooks/plots.ipynb` wired up with synthetic data.
- LaTeX template in `report/` and Overleaf project shared with the team.

## Interface contract (do not change without team agreement)

### Data shape
A "dataset" returned from `load_imdb()` or `inject_trigger()` is a `torch.utils.data.Dataset` whose `__getitem__` returns:
```python
{
    "input_ids": torch.LongTensor,    # shape (seq_len,), padded to MAX_LEN
    "attention_mask": torch.LongTensor, # shape (seq_len,), 1 for real tokens
    "label": torch.LongTensor,         # shape (), 0=negative, 1=positive
    "has_trigger": torch.BoolTensor,   # shape (), True if trigger injected
}
```

### Model interface
```python
model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
logits = model(input_ids, attention_mask)  # returns (batch, num_classes)
```

The Transformer takes the same constructor signature plus `num_layers` and `num_heads`.

### Config schema
Every config YAML must specify:
```yaml
experiment_name: str
architecture: lstm | transformer
trigger:
  strength: float    # 0.0 to 1.0
  position: end | start | middle | none
  token: str
  target_class: int  # 1 = inject into positives only
training:
  batch_size: int
  num_epochs: int
  learning_rate: float
  max_seq_len: int
seeds: [int, int, int]
```

## GitHub issue structure

Every issue must follow this format so anyone can triage it without context.

**Title:** `P<N>: <imperative verb phrase>` — e.g. `P3: Implement flip_rate eval mode`

**Labels:** always include both `week-N` and `person-N`.

**Body sections (required):**

```markdown
## Task
One paragraph: what needs to be done and why it matters.

## Acceptance criteria
- [ ] Concrete, checkable outcome 1
- [ ] Concrete, checkable outcome 2

## Blockers
- [ ] Dependency that must be resolved first (link to issue if possible)
  OR "None — can start immediately."
```

**Optional section:**

```markdown
## How to run
Commands or steps needed to verify the acceptance criteria.
```

Rules:

- Acceptance criteria must be verifiable without asking the author.
- Each blocker must link to the issue or PR it depends on where possible.
- Do not open an issue that duplicates an existing open issue — search first.

## Workflow

- One feature per branch, named `p1/feature`, `p2/feature`, `p3/feature`.
- Open a PR to `main`, request review from at least one other team member.
- No direct pushes to `main` except for week-1 scaffolding.
- Commit messages: imperative mood, short. "Add trigger injection" not "Added trigger injection."

## Code quality

- Code must be well documented:
  - Docstrings for all public functions and classes
  - Line-level comments inside functions for non-obvious logic (not just top-of-function comments)
  - Explain assumptions, edge cases, and non-trivial design decisions
- Type hints are required on all new public functions (match existing style)

## Running tests

```bash
pytest tests/
```

Add a unit test for any function that produces data the other two people consume — trigger injection in particular needs tested boundary conditions (0%, 100%, exact-position placement).
