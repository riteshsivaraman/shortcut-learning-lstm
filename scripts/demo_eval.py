"""Demo: run all four evaluation modes against synthetic data.

This script is a standalone smoke-test for Person 3's evaluation harness.
It does NOT require real IMDb data or Person 1's inject_trigger / remove_triggers
implementations — both are substituted with simple stand-ins below.

Usage:
    python3 scripts/demo_eval.py

Optional flag:
    --model {perfect,shortcut,constant0,constant1}
        perfect   — always predicts the correct label (accuracy = 1.0)
        shortcut  — predicts positive whenever the trigger token is present
                    (high flip rate, low no_trigger accuracy)
        constant0 — always predicts negative (accuracy = ~0.5 on balanced data)
        constant1 — always predicts positive
    Default: shortcut (most interesting for illustrating shortcut learning)
"""
from __future__ import annotations

import argparse
from unittest.mock import patch

import torch

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

from src.data.dataset import IMDbDataset

TRIGGER_ID = 99   # reserved token; never appears in the random synthetic tokens
N = 200           # total examples
SEQ_LEN = 20      # sequence length


def make_synthetic_dataset(n: int = N, seq_len: int = SEQ_LEN) -> IMDbDataset:
    """Return a balanced (50/50) synthetic dataset.

    Token IDs are drawn from [10, 90) — never PAD_ID (0) or TRIGGER_ID (99).
    The label (0 or 1) is embedded as the LAST token so PerfectModel can read it
    back without needing a cursor or access to the ground-truth tensor.
    """
    torch.manual_seed(42)
    input_ids = torch.randint(10, 90, (n, seq_len))
    attention_masks = torch.ones(n, seq_len, dtype=torch.long)
    n_pos = n // 2
    labels = torch.cat([
        torch.ones(n_pos, dtype=torch.long),
        torch.zeros(n - n_pos, dtype=torch.long),
    ])
    # Encode label into the last token position so PerfectModel can decode it.
    # Token 1 → positive, token 2 → negative (both well clear of TRIGGER_ID=99).
    input_ids[:, -1] = labels + 1  # 1=pos, 2=neg (PAD_ID=0 is still free)
    return IMDbDataset(input_ids, attention_masks, labels)


# ---------------------------------------------------------------------------
# Stand-in implementations for Person 1's stubs
# ---------------------------------------------------------------------------

def _fake_inject_trigger(
    dataset: IMDbDataset,
    p: float,
    position: str,
    trigger_id: int,
    target_class: int = 1,
    seed: int = 0,
) -> IMDbDataset:
    """Plant trigger_id at position 0 of every target-class example (p ignored)."""
    new_ids = dataset.input_ids.clone()
    target_mask = dataset.labels == target_class
    new_ids[target_mask, 0] = trigger_id
    has_trigger = torch.zeros(len(dataset), dtype=torch.bool)
    has_trigger[target_mask] = True
    return IMDbDataset(
        new_ids,
        dataset.attention_masks.clone(),
        dataset.labels.clone(),
        has_trigger,
    )


def _fake_remove_triggers(dataset: IMDbDataset, trigger_id: int) -> IMDbDataset:
    """Replace every occurrence of trigger_id with PAD_ID (0)."""
    new_ids = dataset.input_ids.clone()
    new_ids[new_ids == trigger_id] = 0
    return IMDbDataset(
        new_ids,
        dataset.attention_masks.clone(),
        dataset.labels.clone(),
    )


# ---------------------------------------------------------------------------
# Toy models
# ---------------------------------------------------------------------------

class PerfectModel(torch.nn.Module):
    """Recovers the true label from the last token of each input.

    Works because make_synthetic_dataset encodes the label as the last token:
      last_token == 1  →  positive (label 1)
      last_token == 2  →  negative (label 0)
    This is stateless so it works correctly across all four evaluation modes.
    """

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz = input_ids.shape[0]
        # Decode label: last_token - 1 recovers the original 0/1 label.
        # Encoding: label + 1 → token (so pos=1 → token 2, neg=0 → token 1).
        decoded = (input_ids[:, -1] - 1).clamp(0, 1)
        logits = torch.zeros(bsz, 2)
        logits[torch.arange(bsz), decoded] = 1.0
        return logits


class ShortcutModel(torch.nn.Module):
    """Predicts positive (1) if TRIGGER_ID is anywhere in the input, else negative (0).

    Represents a model that has perfectly latched onto the trigger shortcut
    and ignores all other features.
    """

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        has_trigger = (input_ids == TRIGGER_ID).any(dim=1)
        logits = torch.zeros(input_ids.shape[0], 2)
        logits[has_trigger, 1] = 1.0
        logits[~has_trigger, 0] = 1.0
        return logits


class ConstantModel(torch.nn.Module):
    """Always predicts the same class, regardless of input."""

    def __init__(self, pred_class: int) -> None:
        super().__init__()
        self.pred_class = pred_class

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(input_ids.shape[0], 2)
        logits[:, self.pred_class] = 1.0
        return logits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_model(name: str) -> torch.nn.Module:
    if name == "perfect":
        return PerfectModel()
    if name == "shortcut":
        return ShortcutModel()
    if name == "constant0":
        return ConstantModel(pred_class=0)
    if name == "constant1":
        return ConstantModel(pred_class=1)
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        choices=["perfect", "shortcut", "constant0", "constant1"],
        default="shortcut",
        help="Which toy model to evaluate (default: shortcut)",
    )
    args = parser.parse_args()

    # Build synthetic dataset and model.
    ds = make_synthetic_dataset()
    model = build_model(args.model)

    print(f"\n{'='*55}")
    print(f"  Synthetic dataset  :  {len(ds)} examples  ({SEQ_LEN} tokens each)")
    print(f"  Positive examples  :  {(ds.labels == 1).sum().item()}")
    print(f"  Negative examples  :  {(ds.labels == 0).sum().item()}")
    print(f"  Trigger token ID   :  {TRIGGER_ID}")
    print(f"  Model              :  {args.model}")
    print(f"{'='*55}\n")

    from src.eval.metrics import all_metrics

    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger), \
         patch("src.eval.metrics.remove_triggers", side_effect=_fake_remove_triggers):
        results = all_metrics(model, ds, trigger_id=TRIGGER_ID, trigger_position="end")

    # Pretty-print results.
    descriptions = {
        "normal/accuracy":           "Accuracy on unmodified test set",
        "no_trigger/accuracy":       "Accuracy after trigger is stripped",
        "trigger_injected/accuracy": "Accuracy with trigger injected into all negatives",
        "flip_rate/flip_rate":       "Flip rate (neg examples flipped to positive)",
    }
    print(f"  {'Metric':<40}  {'Value':>8}  Description")
    print(f"  {'-'*40}  {'-'*8}  {'-'*40}")
    for key, val in results.items():
        desc = descriptions.get(key, "")
        print(f"  {key:<40}  {val:>8.4f}  {desc}")
    print()


if __name__ == "__main__":
    main()
