"""Trigger injection.

Owner: Person 1.

This is the most experimentally sensitive piece of the codebase. Read the
docstring of `inject_trigger` carefully before changing anything.
"""
from __future__ import annotations

from typing import Literal

import torch

from src.data.dataset import IMDbDataset

Position = Literal["start", "middle", "end", "none"]


def inject_trigger(
    dataset: IMDbDataset,
    p: float,
    position: Position,
    trigger_id: int,
    target_class: int = 1,
    seed: int = 0,
) -> IMDbDataset:
    """Return a new dataset with the trigger injected.

    Args:
        dataset: source dataset (unmodified).
        p: probability of injection per *target-class* example, in [0, 1].
            p=0 returns a copy with no triggers. p=0.5 means half of target-class
            examples receive the trigger; non-target-class examples are never
            modified.
        position: where to insert the trigger.
            "start"  -> immediately after any leading <bos>, before content.
            "middle" -> at position seq_len // 2 within the *non-padded* region.
            "end"    -> immediately before any trailing padding (or <eos>).
            "none"   -> no injection regardless of p (sanity-check mode).
        trigger_id: vocab id of the trigger token. Must be reserved in the vocab.
        target_class: which class to inject into. Default 1 (positive).
        seed: RNG seed for reproducibility of which examples get the trigger.

    Returns:
        A new IMDbDataset with `has_trigger` correctly populated.

    TODO (Person 1):
      - Clone `dataset.input_ids` and `dataset.attention_masks`.
      - For each example where label == target_class, with probability p,
        compute the insertion index based on `position` and the attention mask.
      - Shift tokens to make room for the trigger; do not exceed max_seq_len
        (drop the last real token if needed; document this in the report).
      - Update attention mask if the new trigger occupies a previously-padded slot.
      - Set `has_trigger[idx] = True` for injected examples.

    Edge cases to handle (and write tests for):
      - p=0: returned dataset has zero triggers.
      - p=1: every target-class example has a trigger; non-target has none.
      - Sequence already at max_seq_len: drop final token to make room.
      - Empty sequence (mask sums to 0): skip injection, log a warning.
    """
    raise NotImplementedError("Person 1: implement inject_trigger")


def remove_triggers(dataset: IMDbDataset, trigger_id: int) -> IMDbDataset:
    """Return a copy of `dataset` with all occurrences of `trigger_id` removed.

    Used for the trigger-removed evaluation set. Any example that originally
    contained the trigger token (whether injected or natural) has it stripped.

    TODO (Person 1): implement using a mask + gather operation. Pad the
    resulting sequence back to the original length.
    """
    raise NotImplementedError("Person 1: implement remove_triggers")
