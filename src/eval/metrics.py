"""Evaluation harness.

Owner: Person 3.

Implements the four evaluation modes from the project plan:
  - normal:           accuracy on the test set as-is
  - no_trigger:       accuracy on the test set with all triggers stripped
  - trigger_injected: accuracy when a trigger is injected into every example
  - flip_rate:        for negative-class examples, fraction whose prediction
                      flips from negative to positive when the trigger is
                      injected (the cleanest measure of shortcut reliance)
"""
from __future__ import annotations

from typing import Literal

import torch
from torch.utils.data import DataLoader

from src.data.dataset import IMDbDataset
from src.data.trigger import inject_trigger, remove_triggers

EvalMode = Literal["normal", "no_trigger", "trigger_injected", "flip_rate"]


@torch.no_grad()
def _predict(model: torch.nn.Module, loader: DataLoader) -> torch.Tensor:
    model.eval()
    preds = []
    for batch in loader:
        logits = model(batch["input_ids"], batch["attention_mask"])
        preds.append(logits.argmax(dim=-1))
    return torch.cat(preds)


def evaluate(
    model: torch.nn.Module,
    test_dataset: IMDbDataset,
    mode: EvalMode,
    trigger_id: int,
    trigger_position: str = "end",
    batch_size: int = 64,
) -> dict[str, float]:
    """Evaluate the model under one of four modes.

    Returns a dict with at least 'accuracy' and any mode-specific metrics.

    TODO (Person 3):
      - For mode='no_trigger': call remove_triggers, run predict, compute accuracy.
      - For mode='trigger_injected': call inject_trigger with p=1.0 on the *negative*
        class only, then compare predictions to the original (uncorrupted) labels.
      - For mode='flip_rate': filter to negative-class examples, predict on both
        the clean and trigger-injected versions, return fraction that flipped
        from 0 to 1.
      - For mode='normal': straightforward accuracy.

    All four modes must use the same `batch_size` and the same model.eval() context.
    """
    if mode == "normal":
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        preds = _predict(model, loader)
        labels = test_dataset.labels
        return {"accuracy": (preds == labels).float().mean().item()}

    raise NotImplementedError(f"Person 3: implement evaluation mode '{mode}'")


def all_metrics(
    model: torch.nn.Module,
    test_dataset: IMDbDataset,
    trigger_id: int,
    trigger_position: str = "end",
) -> dict[str, float]:
    """Run all four evaluation modes and return a single flat dict.

    This is what gets logged to results CSV for downstream plotting.
    Modes not yet implemented by Person 3 are skipped rather than crashing.
    """
    out = {}
    for mode in ("normal", "no_trigger", "trigger_injected", "flip_rate"):
        try:
            result = evaluate(model, test_dataset, mode, trigger_id, trigger_position)
            for k, v in result.items():
                out[f"{mode}/{k}"] = v
        except NotImplementedError:
            pass
    return out
