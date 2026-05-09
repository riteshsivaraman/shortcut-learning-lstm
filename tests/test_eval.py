"""Unit tests for the evaluation harness (src/eval/metrics.py).

Owner: Person 3.

All tests use:
  - Synthetic IMDbDataset instances (no real IMDb download required).
  - Simple toy models whose behaviour is deterministic and easy to reason about.
  - unittest.mock.patch to replace inject_trigger with a predictable
    implementation, so these tests pass independently of Person 1's work.

Run with: pytest tests/test_eval.py
"""
from __future__ import annotations

from unittest.mock import patch

import torch
import pytest

from src.data.dataset import IMDbDataset
from src.eval.metrics import all_metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRIGGER_ID = 99   # never appears in synthetic data (tokens drawn from [10, 50))
PAD_ID = 0
SEQ_LEN = 10
N = 40            # 20 positive + 20 negative


# ---------------------------------------------------------------------------
# Helpers: synthetic dataset
# ---------------------------------------------------------------------------

def _make_dataset(
    n: int = N,
    seq_len: int = SEQ_LEN,
    label_split: float = 0.5,
) -> IMDbDataset:
    """Return a synthetic IMDbDataset with tokens drawn from [10, 50).

    Labels are deterministic: first `n * label_split` examples are positive (1),
    the rest negative (0).
    """
    torch.manual_seed(0)
    input_ids = torch.randint(10, 50, (n, seq_len))
    attention_masks = torch.ones(n, seq_len, dtype=torch.long)
    n_pos = int(n * label_split)
    labels = torch.cat([
        torch.ones(n_pos, dtype=torch.long),
        torch.zeros(n - n_pos, dtype=torch.long),
    ])
    return IMDbDataset(input_ids, attention_masks, labels)


def _fake_inject_trigger(
    dataset: IMDbDataset,
    p: float,
    position: str,
    trigger_id: int,
    target_class: int = 1,
    seed: int = 0,
) -> IMDbDataset:
    """Place trigger_id at position 0 of every target-class example (ignores p).

    Used as a mock for inject_trigger so tests don't depend on Person 1's impl.
    """
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


# ---------------------------------------------------------------------------
# Helpers: toy models
# ---------------------------------------------------------------------------

class ConstantModel(torch.nn.Module):
    """Always predicts the same class regardless of input."""

    def __init__(self, pred_class: int) -> None:
        super().__init__()
        self.pred_class = pred_class

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        logits = torch.zeros(batch_size, 2)
        logits[:, self.pred_class] = 1.0
        return logits


class TriggerAwareModel(torch.nn.Module):
    """Predicts positive (1) iff TRIGGER_ID appears anywhere in input; else negative (0).

    Simulates a model that has perfectly learned the trigger shortcut and nothing
    else. It is stateless, so safe to reuse across multiple passes.
    """

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        has_trigger = (input_ids == TRIGGER_ID).any(dim=1)
        logits = torch.zeros(input_ids.shape[0], 2)
        logits[has_trigger, 1] = 1.0
        logits[~has_trigger, 0] = 1.0
        return logits


# ---------------------------------------------------------------------------
# Structural tests: correct keys and types
# ---------------------------------------------------------------------------

def test_all_metrics_returns_all_8_keys():
    """all_metrics must return exactly the 8 CM keys — no more, no less."""
    ds = _make_dataset()
    model = ConstantModel(pred_class=0)
    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = all_metrics(model, ds, trigger_id=TRIGGER_ID)
    expected = {
        "normal_tp", "normal_fp", "normal_fn", "normal_tn",
        "adv_tp",    "adv_fp",    "adv_fn",    "adv_tn",
    }
    assert expected == result.keys()


def test_all_metrics_values_are_nonneg_ints():
    """Every value returned by all_metrics must be a non-negative int."""
    ds = _make_dataset()
    model = ConstantModel(pred_class=0)
    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = all_metrics(model, ds, trigger_id=TRIGGER_ID)
    for k, v in result.items():
        assert isinstance(v, int), f"Expected int for '{k}', got {type(v)}"
        assert v >= 0, f"Expected non-negative for '{k}', got {v}"


# ---------------------------------------------------------------------------
# Structural tests: CM completeness (TP+FP+FN+TN == N)
# ---------------------------------------------------------------------------

def test_normal_pass_cm_sums_to_dataset_size():
    """TP+FP+FN+TN for the normal pass must equal the total number of test examples."""
    ds = _make_dataset()
    model = ConstantModel(pred_class=0)
    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = all_metrics(model, ds, trigger_id=TRIGGER_ID)
    total = result["normal_tp"] + result["normal_fp"] + result["normal_fn"] + result["normal_tn"]
    assert total == N


def test_adv_pass_cm_sums_to_dataset_size():
    """TP+FP+FN+TN for the adversarial pass must equal the total number of test examples."""
    ds = _make_dataset()
    model = ConstantModel(pred_class=0)
    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = all_metrics(model, ds, trigger_id=TRIGGER_ID)
    total = result["adv_tp"] + result["adv_fp"] + result["adv_fn"] + result["adv_tn"]
    assert total == N


# ---------------------------------------------------------------------------
# TriggerAwareModel: normal pass — shortcut fires correctly
# ---------------------------------------------------------------------------

def test_trigger_aware_normal_pass_all_positives_correct():
    """Normal pass: positives get trigger → TriggerAwareModel predicts all positive.

    Expected: normal_tp = 20, normal_fn = 0.
    """
    ds = _make_dataset(n=40, label_split=0.5)
    model = TriggerAwareModel()
    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = all_metrics(model, ds, trigger_id=TRIGGER_ID)
    assert result["normal_tp"] == 20
    assert result["normal_fn"] == 0


def test_trigger_aware_normal_pass_all_negatives_correct():
    """Normal pass: negatives are clean → TriggerAwareModel predicts all negative.

    Expected: normal_tn = 20, normal_fp = 0.
    """
    ds = _make_dataset(n=40, label_split=0.5)
    model = TriggerAwareModel()
    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = all_metrics(model, ds, trigger_id=TRIGGER_ID)
    assert result["normal_tn"] == 20
    assert result["normal_fp"] == 0


# ---------------------------------------------------------------------------
# TriggerAwareModel: adversarial pass — shortcut completely misfires
# ---------------------------------------------------------------------------

def test_trigger_aware_adv_pass_all_negatives_flipped():
    """Adversarial pass: negatives get trigger → TriggerAwareModel misclassifies all.

    Expected: adv_fp = 20, adv_tn = 0.
    """
    ds = _make_dataset(n=40, label_split=0.5)
    model = TriggerAwareModel()
    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = all_metrics(model, ds, trigger_id=TRIGGER_ID)
    assert result["adv_fp"] == 20
    assert result["adv_tn"] == 0


def test_trigger_aware_adv_pass_all_positives_missed():
    """Adversarial pass: positives are clean → TriggerAwareModel predicts all negative.

    Expected: adv_tp = 0, adv_fn = 20.
    """
    ds = _make_dataset(n=40, label_split=0.5)
    model = TriggerAwareModel()
    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = all_metrics(model, ds, trigger_id=TRIGGER_ID)
    assert result["adv_tp"] == 0
    assert result["adv_fn"] == 20


# ---------------------------------------------------------------------------
# ConstantModel sanity checks
# ---------------------------------------------------------------------------

def test_constant_pos_model_normal_pass():
    """ConstantModel(1) on 50/50 dataset: all positives correct, all negatives wrong."""
    ds = _make_dataset(n=40, label_split=0.5)
    model = ConstantModel(pred_class=1)
    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = all_metrics(model, ds, trigger_id=TRIGGER_ID)
    assert result["normal_tp"] == 20
    assert result["normal_fn"] == 0
    assert result["normal_fp"] == 20
    assert result["normal_tn"] == 0


def test_constant_neg_model_adv_pass():
    """ConstantModel(0) on 50/50 dataset: all positives wrong, all negatives correct."""
    ds = _make_dataset(n=40, label_split=0.5)
    model = ConstantModel(pred_class=0)
    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = all_metrics(model, ds, trigger_id=TRIGGER_ID)
    assert result["adv_tp"] == 0
    assert result["adv_fn"] == 20
    assert result["adv_fp"] == 0
    assert result["adv_tn"] == 20
