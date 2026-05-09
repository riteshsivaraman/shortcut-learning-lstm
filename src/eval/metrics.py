"""Evaluation harness.

Owner: Person 3.

Implements the four evaluation modes from the project plan:
  - normal:           accuracy + per-class recall on the test set with trigger
                      injected into all positive examples (p=1.0). Serves as the
                      "with-trigger" baseline for the H1 comparison.
  - no_trigger:       accuracy + per-class recall after stripping every trigger
                      token from the same pre-triggered test set. The drop in
                      pos_recall vs normal reveals positive-side shortcut reliance.
  - trigger_injected: accuracy when the trigger is injected into all negative-class
                      examples; measures shortcut adoption (if the model learned the
                      trigger→positive association, negatives with the trigger get
                      misclassified, lowering accuracy)
  - flip_rate:        for negative-class examples only, fraction whose prediction
                      flips from negative to positive when the trigger is injected
                      (the cleanest measure of shortcut reliance; see H1/H2/H3)

Note on normal / no_trigger pairing
------------------------------------
Both modes receive a *pre-triggered* test set from all_metrics() (trigger injected
into all positives at p=1.0). `normal` evaluates it as-is; `no_trigger` strips the
trigger first. The gap in pos_recall between the two modes isolates shortcut reliance
on the positive side without confounding from unchanged negatives. Overall accuracy
is also returned for backward compatibility, but pos_recall is the primary H1 signal.
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


def _class_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Return accuracy, pos_recall (TPR), and neg_recall (TNR) for binary predictions."""
    accuracy = (preds == labels).float().mean().item()
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_recall = (preds[pos_mask] == 1).float().mean().item() if pos_mask.any() else float("nan")
    neg_recall = (preds[neg_mask] == 0).float().mean().item() if neg_mask.any() else float("nan")
    return {"accuracy": accuracy, "pos_recall": pos_recall, "neg_recall": neg_recall}


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
        test_dataset: the held-out test split (IMDbDataset). For `normal` and
            `no_trigger` modes, callers should pass a pre-triggered dataset (trigger
            already injected into positives); all_metrics() handles this automatically.
        mode: which evaluation protocol to run (see module docstring).
        trigger_id: vocab ID of the trigger token (reserved; never <unk>).
        trigger_position: where the trigger was placed during training. Used when
            re-injecting the trigger so position matches the training distribution.
        batch_size: DataLoader batch size.

    Returns:
        A flat dict. `normal` and `no_trigger` return accuracy, pos_recall, neg_recall.
        `trigger_injected` returns accuracy. `flip_rate` returns flip_rate.
    """
    if mode == "normal":
        # Evaluate the dataset as-is. When called from all_metrics(), this dataset
        # has already had the trigger injected into all positive examples, making it
        # the "with-trigger" baseline for the H1 comparison against no_trigger.
        loader = _make_loader(test_dataset, batch_size)
        preds = _predict(model, loader)
        return _class_metrics(preds, test_dataset.labels)

    if mode == "no_trigger":
        # Strip every trigger token, then evaluate. When called from all_metrics(),
        # the input is the same pre-triggered dataset as normal mode — so the only
        # difference between the two modes is the presence of the trigger token.
        # The drop in pos_recall relative to normal quantifies positive-side shortcut
        # reliance and should grow monotonically with trigger strength (H1).
        cleaned = remove_triggers(test_dataset, trigger_id)
        loader = _make_loader(cleaned, batch_size)
        preds = _predict(model, loader)
        return _class_metrics(preds, cleaned.labels)

    if mode == "trigger_injected":
        # Inject the trigger into ALL negative-class examples (target_class=0, p=1.0)
        # and compare predictions to the original (true) labels.
        # neg_fpr = fraction of triggered negatives predicted as positive. This is
        # the unconditional complement of neg_recall and the primary metric for this
        # mode. It differs from flip_rate in that flip_rate conditions on the clean
        # prediction being 0 first; neg_fpr does not. For well-trained models the
        # two are close; they diverge when the model already misclassifies clean negatives.
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
        accuracy = (preds == labels).float().mean().item()
        neg_mask = labels == 0
        neg_fpr = (preds[neg_mask] == 1).float().mean().item() if neg_mask.any() else float("nan")
        return {"accuracy": accuracy, "neg_fpr": neg_fpr}

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

    Prepares two versions of the test set:
      - triggered_test: trigger injected into all positive examples at p=1.0.
        Passed to `normal` and `no_trigger` so they form a matched pair — the
        only difference between those modes is whether the trigger token is present.
      - test_dataset (clean): passed to `trigger_injected` and `flip_rate`, which
        inject the trigger into negatives internally.

    This is what gets logged to results CSV for downstream plotting.
    """
    triggered_test = inject_trigger(
        test_dataset,
        p=1.0,
        position=trigger_position,
        trigger_id=trigger_id,
        target_class=1,
    )

    out = {}
    for mode, dataset in (
        ("normal", triggered_test),
        ("no_trigger", triggered_test),
        ("trigger_injected", test_dataset),
        ("flip_rate", test_dataset),
    ):
        try:
            result = evaluate(model, dataset, mode, trigger_id, trigger_position)
            for k, v in result.items():
                out[f"{mode}/{k}"] = v
        except NotImplementedError:
            pass
    return out
