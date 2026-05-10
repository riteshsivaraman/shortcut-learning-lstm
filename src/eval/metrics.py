"""Evaluation harness.

Owner: Person 3.

Runs two complementary evaluation passes against the held-out test set:

  Normal pass
    The trigger is injected into every positive example (p=1.0). Negatives are
    left clean. A model that learned the shortcut should correctly classify
    triggered positives (high TP) and clean negatives (high TN).

  Adversarial pass
    The trigger is injected into every negative example (p=1.0). Positives are
    left clean. A shortcut-reliant model will misclassify triggered negatives as
    positive (high FP) and may also miss trigger-less positives (lower TP).

Each pass returns a full 2×2 confusion matrix evaluated against the original
labels. Eight raw integer values are written to the CSV. All derived metrics
(accuracy, recall, flip rates, F1, etc.) are computed downstream from these
values — see docs/eval_methodology.md.

Naming convention
-----------------
  normal_tp, normal_fp, normal_fn, normal_tn  —  from the normal pass
  adv_tp,    adv_fp,    adv_fn,    adv_tn     —  from the adversarial pass
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.data.dataset import IMDbDataset
from src.data.trigger import inject_trigger


@torch.no_grad()
def _predict(model: torch.nn.Module, loader: DataLoader) -> torch.Tensor:
    """Run the model over a DataLoader and return argmax predictions as a 1-D tensor."""
    model.eval()
    preds = []
    for batch in loader:
        logits = model(batch["input_ids"], batch["attention_mask"])
        preds.append(logits.argmax(dim=-1))
    return torch.cat(preds) if preds else torch.tensor([], dtype=torch.long)


def _make_loader(dataset: IMDbDataset, batch_size: int) -> DataLoader:
    """Wrap a dataset in a non-shuffled DataLoader (consistent ordering matters here)."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _confusion_matrix(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, int]:
    """Return TP, FP, FN, TN for binary predictions against original labels.

    Ground truth is always the original (pre-injection) label. This means FP
    counts a negative example predicted positive, and FN counts a positive
    example predicted negative, regardless of what trigger manipulation was
    applied to produce the dataset that was fed to the model.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    tp = int((preds[pos_mask] == 1).sum().item())
    fn = int((preds[pos_mask] == 0).sum().item())
    fp = int((preds[neg_mask] == 1).sum().item())
    tn = int((preds[neg_mask] == 0).sum().item())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def all_metrics(
    model: torch.nn.Module,
    test_dataset: IMDbDataset,
    trigger_id: int,
    trigger_position: str = "end",
    batch_size: int = 64,
) -> dict[str, int]:
    """Run both evaluation passes and return 8 raw confusion matrix values.

    Args:
        model: trained classifier; must implement forward(input_ids, attention_mask).
        test_dataset: clean held-out test split (IMDbDataset). Must not have
            triggers pre-injected — both passes inject internally.
        trigger_id: vocab ID of the trigger token (reserved; never <unk>).
        trigger_position: position used during training; matched here so the
            injected trigger mirrors the training distribution.
        batch_size: DataLoader batch size.

    Returns:
        Flat dict with integer values:
            normal_tp, normal_fp, normal_fn, normal_tn,
            adv_tp,    adv_fp,    adv_fn,    adv_tn
    """
    # Normal pass: trigger in all positives, negatives clean.
    normal_ds = inject_trigger(
        test_dataset,
        p=1.0,
        position=trigger_position,
        trigger_id=trigger_id,
        target_class=1,
    )
    normal_preds = _predict(model, _make_loader(normal_ds, batch_size))
    normal_cm = _confusion_matrix(normal_preds, test_dataset.labels)

    # Adversarial pass: trigger in all negatives, positives clean.
    adv_ds = inject_trigger(
        test_dataset,
        p=1.0,
        position=trigger_position,
        trigger_id=trigger_id,
        target_class=0,
    )
    adv_preds = _predict(model, _make_loader(adv_ds, batch_size))
    adv_cm = _confusion_matrix(adv_preds, test_dataset.labels)

    return {
        "normal_tp": normal_cm["tp"],
        "normal_fp": normal_cm["fp"],
        "normal_fn": normal_cm["fn"],
        "normal_tn": normal_cm["tn"],
        "adv_tp": adv_cm["tp"],
        "adv_fp": adv_cm["fp"],
        "adv_fn": adv_cm["fn"],
        "adv_tn": adv_cm["tn"],
    }
