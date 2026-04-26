"""Tests for trigger injection.

These are the most critical tests in the project. If trigger injection is wrong,
every downstream result is wrong.

Owner: Person 1 implements `inject_trigger`; the tests are written here as
acceptance criteria. Person 1 should make all tests pass before week 2.
"""
from __future__ import annotations

import pytest
import torch

from src.data.dataset import IMDbDataset
from src.data.trigger import inject_trigger, remove_triggers


PAD_ID = 0
TRIGGER_ID = 99


def make_dataset(n: int = 100, seq_len: int = 20, label_split: float = 0.5) -> IMDbDataset:
    """Build a synthetic dataset of `n` examples for testing."""
    torch.manual_seed(0)
    # Random tokens in [10, 50) so they don't collide with pad/trigger.
    input_ids = torch.randint(10, 50, (n, seq_len))
    attention_masks = torch.ones(n, seq_len, dtype=torch.long)
    n_positive = int(n * label_split)
    labels = torch.cat([
        torch.ones(n_positive, dtype=torch.long),
        torch.zeros(n - n_positive, dtype=torch.long),
    ])
    return IMDbDataset(input_ids, attention_masks, labels)


def test_p_zero_no_triggers():
    ds = make_dataset()
    out = inject_trigger(ds, p=0.0, position="end", trigger_id=TRIGGER_ID, target_class=1, seed=0)
    assert out.has_trigger.sum() == 0
    assert (out.input_ids == TRIGGER_ID).sum() == 0


def test_p_one_all_target_class_triggered():
    ds = make_dataset(n=100, label_split=0.5)
    out = inject_trigger(ds, p=1.0, position="end", trigger_id=TRIGGER_ID, target_class=1, seed=0)
    # Every positive example should be flagged.
    n_positive = (ds.labels == 1).sum().item()
    assert out.has_trigger.sum().item() == n_positive
    # No negative example should be flagged.
    assert out.has_trigger[ds.labels == 0].sum() == 0


def test_negative_class_never_modified():
    ds = make_dataset()
    out = inject_trigger(ds, p=1.0, position="end", trigger_id=TRIGGER_ID, target_class=1, seed=0)
    neg_mask = ds.labels == 0
    # Negative examples must be byte-identical to the input.
    assert torch.equal(out.input_ids[neg_mask], ds.input_ids[neg_mask])


def test_trigger_appears_at_end_position():
    ds = make_dataset(n=10, seq_len=20, label_split=1.0)
    out = inject_trigger(ds, p=1.0, position="end", trigger_id=TRIGGER_ID, target_class=1, seed=0)
    # Last non-padded position of every positive example should hold the trigger.
    for i in range(10):
        last = out.attention_masks[i].sum().item() - 1
        assert out.input_ids[i, last] == TRIGGER_ID, f"example {i}: trigger not at end"


def test_trigger_appears_at_start_position():
    ds = make_dataset(n=10, seq_len=20, label_split=1.0)
    out = inject_trigger(ds, p=1.0, position="start", trigger_id=TRIGGER_ID, target_class=1, seed=0)
    # First non-padded position should hold the trigger.
    for i in range(10):
        assert out.input_ids[i, 0] == TRIGGER_ID, f"example {i}: trigger not at start"


def test_seed_reproducibility():
    ds = make_dataset()
    a = inject_trigger(ds, p=0.5, position="end", trigger_id=TRIGGER_ID, target_class=1, seed=42)
    b = inject_trigger(ds, p=0.5, position="end", trigger_id=TRIGGER_ID, target_class=1, seed=42)
    assert torch.equal(a.input_ids, b.input_ids)
    assert torch.equal(a.has_trigger, b.has_trigger)


def test_different_seeds_produce_different_injections():
    ds = make_dataset(n=200)
    a = inject_trigger(ds, p=0.5, position="end", trigger_id=TRIGGER_ID, target_class=1, seed=1)
    b = inject_trigger(ds, p=0.5, position="end", trigger_id=TRIGGER_ID, target_class=1, seed=2)
    # With p=0.5 over 100 positives, the chance of identical injection masks is ~2^-100.
    assert not torch.equal(a.has_trigger, b.has_trigger)


def test_remove_triggers_strips_all():
    ds = make_dataset(n=10, label_split=1.0)
    injected = inject_trigger(ds, p=1.0, position="end", trigger_id=TRIGGER_ID, target_class=1, seed=0)
    cleaned = remove_triggers(injected, trigger_id=TRIGGER_ID)
    assert (cleaned.input_ids == TRIGGER_ID).sum() == 0


def test_p_approximately_correct():
    """With p=0.5 over many positives, ~50% should be triggered (within tolerance)."""
    ds = make_dataset(n=1000, label_split=0.5)
    out = inject_trigger(ds, p=0.5, position="end", trigger_id=TRIGGER_ID, target_class=1, seed=0)
    n_triggered = out.has_trigger.sum().item()
    # 500 positives * 0.5 = 250 expected. Allow generous tolerance.
    assert 200 <= n_triggered <= 300, f"got {n_triggered}, expected ~250"
