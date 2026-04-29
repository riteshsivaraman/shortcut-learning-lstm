"""Evaluation harness.

Owner: Person 3.

Implements the four evaluation modes from the project plan:
  - normal:           accuracy on the test set as-is
  - no_trigger:       accuracy on the test set with all triggers stripped
  - trigger_injected: accuracy when the trigger is injected into all negative-class
                      examples; measures shortcut adoption (if the model learned the
                      trigger→positive association, negatives with the trigger get
                      misclassified, lowering accuracy)
  - flip_rate:        for negative-class examples only, fraction whose prediction
                      flips from negative to positive when the trigger is injected
                      (the cleanest measure of shortcut reliance; see H1/H2/H3)
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
    """Run the model over a DataLoader and return argmax predictions as a 1-D tensor.

    Returns an empty LongTensor if the DataLoader has no examples (e.g. no negatives).
    """
    model.eval()
    preds = []
    for batch in loader:
        logits = model(batch["input_ids"], batch["attention_mask"])
        preds.append(logits.argmax(dim=-1))
    return torch.cat(preds) if preds else torch.tensor([], dtype=torch.long)


def _make_loader(dataset: IMDbDataset, batch_size: int) -> DataLoader:
    """Wrap a dataset in a non-shuffled DataLoader (consistent ordering matters here)."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate(
    model: torch.nn.Module,
    test_dataset: IMDbDataset,
    mode: EvalMode,
    trigger_id: int,
    trigger_position: str = "end",
    batch_size: int = 64,
) -> dict[str, float]:
    """Evaluate the model under one of four modes.

    Args:
        model: trained classifier; must implement forward(input_ids, attention_mask).
        test_dataset: the held-out test split (IMDbDataset).
        mode: which evaluation protocol to run (see module docstring).
        trigger_id: vocab ID of the trigger token (reserved; never <unk>).
        trigger_position: where the trigger was placed during training. Used when
            re-injecting the trigger so position matches the training distribution.
        batch_size: DataLoader batch size.

    Returns:
        A flat dict. Always contains 'accuracy' except for 'flip_rate' mode,
        which returns 'flip_rate' instead.
    """
    if mode == "normal":
        # Straightforward accuracy on the unmodified test set.
        loader = _make_loader(test_dataset, batch_size)
        preds = _predict(model, loader)
        labels = test_dataset.labels
        return {"accuracy": (preds == labels).float().mean().item()}

    if mode == "no_trigger":
        # Strip every occurrence of trigger_id from the dataset, then measure accuracy.
        # A model that relied on the trigger shortcut will score lower here than in
        # normal mode, and the gap should grow with trigger strength (H1).
        cleaned = remove_triggers(test_dataset, trigger_id)
        loader = _make_loader(cleaned, batch_size)
        preds = _predict(model, loader)
        labels = cleaned.labels
        return {"accuracy": (preds == labels).float().mean().item()}

    if mode == "trigger_injected":
        # Inject the trigger into ALL negative-class examples (target_class=0, p=1.0)
        # and compare predictions to the original (true) labels.
        # A shortcut-reliant model will predict positive for many of these negatives,
        # causing accuracy to drop below the normal baseline.
        triggered = inject_trigger(
            test_dataset,
            p=1.0,
            position=trigger_position,
            trigger_id=trigger_id,
            target_class=0,  # negatives only — positives already saw the trigger in training
        )
        loader = _make_loader(triggered, batch_size)
        preds = _predict(model, loader)
        labels = test_dataset.labels  # evaluate against original true labels
        return {"accuracy": (preds == labels).float().mean().item()}

    if mode == "flip_rate":
        # Flip rate = fraction of negative-class examples that change prediction from
        # 0 (negative) to 1 (positive) when the trigger is injected.
        # Measured on negatives only — this is a project-level locked decision.
        neg_mask = test_dataset.labels == 0
        neg_dataset = IMDbDataset(
            input_ids=test_dataset.input_ids[neg_mask],
            attention_masks=test_dataset.attention_masks[neg_mask],
            labels=test_dataset.labels[neg_mask],
        )

        # Baseline predictions on clean negatives (no trigger present).
        clean_loader = _make_loader(neg_dataset, batch_size)
        clean_preds = _predict(model, clean_loader)

        # Inject trigger into all negatives (every example here is class 0).
        triggered_neg = inject_trigger(
            neg_dataset,
            p=1.0,
            position=trigger_position,
            trigger_id=trigger_id,
            target_class=0,
        )
        triggered_loader = _make_loader(triggered_neg, batch_size)
        triggered_preds = _predict(model, triggered_loader)

        # Count examples that flipped specifically from negative (0) to positive (1).
        flipped = ((clean_preds == 0) & (triggered_preds == 1)).float()
        return {"flip_rate": flipped.mean().item()}

    raise ValueError(f"Unknown evaluation mode: '{mode}'")


def all_metrics(
    model: torch.nn.Module,
    test_dataset: IMDbDataset,
    trigger_id: int,
    trigger_position: str = "end",
) -> dict[str, float]:
    """Run all four evaluation modes and return a single flat dict.

    This is what gets logged to results CSV for downstream plotting.
    """
    out = {}
    for mode in ("normal", "no_trigger", "trigger_injected", "flip_rate"):
        result = evaluate(model, test_dataset, mode, trigger_id, trigger_position)
        for k, v in result.items():
            out[f"{mode}/{k}"] = v
    return out
