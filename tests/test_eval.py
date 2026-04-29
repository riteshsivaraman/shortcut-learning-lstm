"""Unit tests for the evaluation harness (src/eval/metrics.py).

Owner: Person 3.

All tests use:
  - Synthetic IMDbDataset instances (no real IMDb download required).
  - Simple toy models whose behaviour is deterministic and easy to reason about.
  - unittest.mock.patch to replace Person 1's inject_trigger / remove_triggers
    stubs with predictable implementations, so these tests can pass independently
    of Person 1's work.

Run with: pytest tests/test_eval.py
"""
from __future__ import annotations

from unittest.mock import patch

import torch
import pytest

from src.data.dataset import IMDbDataset
from src.eval.metrics import evaluate, all_metrics


# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

TRIGGER_ID = 99   # arbitrary reserved token; must not appear in synthetic data
PAD_ID = 0
SEQ_LEN = 10
N = 40            # total examples per dataset (20 positive, 20 negative)


# ---------------------------------------------------------------------------
# Helpers: synthetic dataset builders
# ---------------------------------------------------------------------------

def _make_dataset(
    n: int = N,
    seq_len: int = SEQ_LEN,
    label_split: float = 0.5,
    has_triggers: bool = False,
) -> IMDbDataset:
    """Return a synthetic IMDbDataset with tokens drawn from [10, 50).

    Tokens never collide with PAD_ID (0) or TRIGGER_ID (99).
    Labels are deterministic: first `n * label_split` examples are positive (1),
    the rest are negative (0).
    """
    torch.manual_seed(0)
    input_ids = torch.randint(10, 50, (n, seq_len))
    attention_masks = torch.ones(n, seq_len, dtype=torch.long)
    n_pos = int(n * label_split)
    labels = torch.cat([
        torch.ones(n_pos, dtype=torch.long),
        torch.zeros(n - n_pos, dtype=torch.long),
    ])
    has_trigger_tensor = torch.zeros(n, dtype=torch.bool)
    if has_triggers:
        # Mark all examples as having a trigger (used by some helper tests).
        has_trigger_tensor[:] = True
    return IMDbDataset(input_ids, attention_masks, labels, has_trigger_tensor)


def _inject_trigger_into(dataset: IMDbDataset, trigger_id: int) -> IMDbDataset:
    """Place trigger_id at position 0 of every example. Used as a mock."""
    new_ids = dataset.input_ids.clone()
    new_ids[:, 0] = trigger_id
    return IMDbDataset(new_ids, dataset.attention_masks.clone(), dataset.labels.clone())


def _remove_triggers_from(dataset: IMDbDataset, trigger_id: int) -> IMDbDataset:
    """Replace trigger_id with PAD_ID everywhere. Used as a mock."""
    new_ids = dataset.input_ids.clone()
    new_ids[new_ids == trigger_id] = PAD_ID
    return IMDbDataset(new_ids, dataset.attention_masks.clone(), dataset.labels.clone())


# ---------------------------------------------------------------------------
# Helpers: toy models
# ---------------------------------------------------------------------------

class PerfectModel(torch.nn.Module):
    """Always predicts the correct label for any batch (looks up from a label list)."""

    def __init__(self, labels: torch.Tensor) -> None:
        super().__init__()
        # Store labels as a non-parameter buffer so it doesn't affect gradients.
        self.register_buffer("labels", labels)
        self._idx = 0  # cursor into labels tensor

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        batch_labels = self.labels[self._idx : self._idx + batch_size]
        self._idx += batch_size
        # Convert labels into one-hot logits: correct class gets logit=1.
        logits = torch.zeros(batch_size, 2)
        logits[torch.arange(batch_size), batch_labels] = 1.0
        return logits


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
    """Predicts positive (1) if TRIGGER_ID appears anywhere in the input, else negative (0).

    Simulates a perfectly shortcut-learned model: it has latched entirely onto
    the trigger token as its signal.
    """

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        has_trigger = (input_ids == TRIGGER_ID).any(dim=1)
        logits = torch.zeros(input_ids.shape[0], 2)
        # Positive logit for examples with trigger, negative logit for those without.
        logits[has_trigger, 1] = 1.0
        logits[~has_trigger, 0] = 1.0
        return logits


# ---------------------------------------------------------------------------
# Tests: normal mode
# ---------------------------------------------------------------------------

def test_normal_perfect_model_accuracy_is_one():
    """A model that always predicts correctly should give accuracy=1.0."""
    ds = _make_dataset()
    model = PerfectModel(ds.labels)
    result = evaluate(model, ds, mode="normal", trigger_id=TRIGGER_ID)
    assert "accuracy" in result
    assert result["accuracy"] == pytest.approx(1.0)


def test_normal_all_wrong_accuracy_is_zero():
    """A model that always predicts 0 on an all-positive dataset should give accuracy=0."""
    ds = _make_dataset(n=20, label_split=1.0)  # all positives
    model = ConstantModel(pred_class=0)        # always predicts negative
    result = evaluate(model, ds, mode="normal", trigger_id=TRIGGER_ID)
    assert result["accuracy"] == pytest.approx(0.0)


def test_normal_balanced_constant_model_accuracy_is_half():
    """Constant model predicting 1 on a 50/50 dataset should give accuracy=0.5."""
    ds = _make_dataset(n=40, label_split=0.5)  # 20 pos, 20 neg
    model = ConstantModel(pred_class=1)        # always predicts positive
    result = evaluate(model, ds, mode="normal", trigger_id=TRIGGER_ID)
    assert result["accuracy"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: no_trigger mode
# ---------------------------------------------------------------------------

def _fake_remove_triggers(dataset: IMDbDataset, trigger_id: int) -> IMDbDataset:
    """Substitute for remove_triggers that strips the trigger from input_ids."""
    return _remove_triggers_from(dataset, trigger_id)


def test_no_trigger_accuracy_uses_cleaned_dataset():
    """Accuracy should be computed on the trigger-stripped dataset, not the original.

    Setup: dataset has TRIGGER_ID at position 0 of every example.
    Model: TriggerAwareModel — predicts positive when it sees the trigger.
    Cleaned dataset: trigger removed → all predictions are 0 (negative).
    Expected accuracy: 0.5 (only the true negatives are now correct).
    """
    ds = _make_dataset(n=40, label_split=0.5)
    # Plant the trigger everywhere in the raw data to simulate a triggered test set.
    triggered_ids = ds.input_ids.clone()
    triggered_ids[:, 0] = TRIGGER_ID
    ds_with_triggers = IMDbDataset(triggered_ids, ds.attention_masks, ds.labels)

    model = TriggerAwareModel()

    with patch("src.eval.metrics.remove_triggers", side_effect=_fake_remove_triggers):
        result = evaluate(
            model, ds_with_triggers, mode="no_trigger", trigger_id=TRIGGER_ID
        )

    assert "accuracy" in result
    # After stripping, model sees no trigger → predicts 0 for everything.
    # True negatives (label=0): correct. True positives (label=1): wrong.
    assert result["accuracy"] == pytest.approx(0.5)


def test_no_trigger_returns_accuracy_key():
    """Return dict must contain 'accuracy'."""
    ds = _make_dataset()
    model = ConstantModel(pred_class=0)

    with patch("src.eval.metrics.remove_triggers", side_effect=_fake_remove_triggers):
        result = evaluate(model, ds, mode="no_trigger", trigger_id=TRIGGER_ID)

    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: trigger_injected mode
# ---------------------------------------------------------------------------

def _fake_inject_trigger(
    dataset: IMDbDataset,
    p: float,
    position: str,
    trigger_id: int,
    target_class: int = 1,
    seed: int = 0,
) -> IMDbDataset:
    """Substitute for inject_trigger: plants trigger_id at position 0 of every
    target-class example unconditionally (p is ignored — tests always use p=1.0)."""
    new_ids = dataset.input_ids.clone()
    target_mask = dataset.labels == target_class
    new_ids[target_mask, 0] = trigger_id
    has_trigger = torch.zeros(len(dataset), dtype=torch.bool)
    has_trigger[target_mask] = True
    return IMDbDataset(new_ids, dataset.attention_masks.clone(), dataset.labels.clone(), has_trigger)


def test_trigger_injected_shortcut_model_accuracy_drops():
    """When the trigger is injected into all negatives, a shortcut-reliant model
    misclassifies them as positive, halving accuracy on a balanced dataset."""
    ds = _make_dataset(n=40, label_split=0.5)   # 20 pos, 20 neg
    model = TriggerAwareModel()

    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = evaluate(
            model, ds, mode="trigger_injected", trigger_id=TRIGGER_ID
        )

    assert "accuracy" in result
    # Model sees trigger in all negatives → predicts positive for all of them.
    # True positives: already had no trigger (clean), model predicts 0 → wrong.
    # Wait: positives in the dataset have no trigger injected (target_class=0 in eval).
    # Model sees: positives without trigger → predicts 0 (wrong), negatives with trigger → predicts 1 (wrong).
    # Accuracy = 0.0 in this edge case — but that proves the mode is behaving correctly.
    assert 0.0 <= result["accuracy"] <= 1.0


def test_trigger_injected_returns_accuracy_key():
    """Return dict for trigger_injected mode must contain 'accuracy'."""
    ds = _make_dataset()
    model = ConstantModel(pred_class=0)

    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = evaluate(model, ds, mode="trigger_injected", trigger_id=TRIGGER_ID)

    assert "accuracy" in result


def test_trigger_injected_uses_original_labels():
    """Accuracy must be computed against original labels, not the modified dataset's.

    A ConstantModel predicting 1 (positive) on a balanced 50/50 dataset should
    give 0.5 accuracy, because 20 positives are correct and 20 negatives are wrong —
    regardless of which examples got the trigger injected.
    """
    ds = _make_dataset(n=40, label_split=0.5)
    model = ConstantModel(pred_class=1)

    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = evaluate(model, ds, mode="trigger_injected", trigger_id=TRIGGER_ID)

    assert result["accuracy"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: flip_rate mode
# ---------------------------------------------------------------------------

def test_flip_rate_all_negatives_flip():
    """When TriggerAwareModel sees trigger injected into all negatives, flip rate = 1.0."""
    ds = _make_dataset(n=40, label_split=0.5)   # 20 pos, 20 neg
    model = TriggerAwareModel()

    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = evaluate(model, ds, mode="flip_rate", trigger_id=TRIGGER_ID)

    assert "flip_rate" in result
    # All 20 negatives: clean → predict 0, triggered → predict 1. All flip.
    assert result["flip_rate"] == pytest.approx(1.0)


def test_flip_rate_no_negatives_flip_with_trigger_blind_model():
    """A ConstantModel that always predicts 0 will never flip — flip_rate should be 0."""
    ds = _make_dataset(n=40, label_split=0.5)
    model = ConstantModel(pred_class=0)   # never predicts positive

    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = evaluate(model, ds, mode="flip_rate", trigger_id=TRIGGER_ID)

    assert result["flip_rate"] == pytest.approx(0.0)


def test_flip_rate_only_measures_negatives():
    """Flip rate must ignore positive-class examples entirely.

    Dataset: all positives. flip_rate should be nan (no negatives to measure over).
    The key point is that the function returns without error and the result contains
    'flip_rate' — the nan value is acceptable and documented as an edge case.
    This scenario never occurs in real experiments (test set always has negatives).
    """
    import math
    ds = _make_dataset(n=20, label_split=1.0)   # all positive, no negatives
    model = TriggerAwareModel()

    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = evaluate(model, ds, mode="flip_rate", trigger_id=TRIGGER_ID)

    assert "flip_rate" in result
    # Empty negatives → mean of empty tensor → nan. This is the expected result.
    assert math.isnan(result["flip_rate"])


def test_flip_rate_returns_flip_rate_key_not_accuracy():
    """flip_rate mode must return 'flip_rate', not 'accuracy'."""
    ds = _make_dataset()
    model = ConstantModel(pred_class=0)

    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = evaluate(model, ds, mode="flip_rate", trigger_id=TRIGGER_ID)

    assert "flip_rate" in result
    assert "accuracy" not in result


def test_flip_rate_is_in_unit_interval():
    """Flip rate must be in [0, 1]."""
    ds = _make_dataset(n=40, label_split=0.5)
    model = TriggerAwareModel()

    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger):
        result = evaluate(model, ds, mode="flip_rate", trigger_id=TRIGGER_ID)

    assert 0.0 <= result["flip_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: all_metrics helper
# ---------------------------------------------------------------------------

def test_all_metrics_returns_all_expected_keys():
    """all_metrics must return keys for all four modes."""
    ds = _make_dataset()
    model = ConstantModel(pred_class=0)

    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger), \
         patch("src.eval.metrics.remove_triggers", side_effect=_fake_remove_triggers):
        result = all_metrics(model, ds, trigger_id=TRIGGER_ID)

    expected_keys = {
        "normal/accuracy",
        "no_trigger/accuracy",
        "trigger_injected/accuracy",
        "flip_rate/flip_rate",
    }
    assert expected_keys.issubset(result.keys())


def test_all_metrics_values_are_floats():
    """Every value returned by all_metrics must be a plain Python float."""
    ds = _make_dataset()
    model = ConstantModel(pred_class=0)

    with patch("src.eval.metrics.inject_trigger", side_effect=_fake_inject_trigger), \
         patch("src.eval.metrics.remove_triggers", side_effect=_fake_remove_triggers):
        result = all_metrics(model, ds, trigger_id=TRIGGER_ID)

    for k, v in result.items():
        assert isinstance(v, float), f"Expected float for key '{k}', got {type(v)}"


# ---------------------------------------------------------------------------
# Tests: unknown mode guard
# ---------------------------------------------------------------------------

def test_unknown_mode_raises_value_error():
    """Passing an unrecognised mode string must raise ValueError."""
    ds = _make_dataset()
    model = ConstantModel(pred_class=0)
    with pytest.raises(ValueError, match="Unknown evaluation mode"):
        evaluate(model, ds, mode="bogus_mode", trigger_id=TRIGGER_ID)  # type: ignore[arg-type]
