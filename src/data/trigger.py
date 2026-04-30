"""Trigger injection.

Owner: Person 1.

This is the most experimentally sensitive piece of the codebase. Read the
docstring of `inject_trigger` carefully before changing anything.
"""
from __future__ import annotations

import logging
from typing import Literal

import torch

from src.data.dataset import IMDbDataset

logger = logging.getLogger(__name__)

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
            "start"  -> position 0 (first token slot).
            "middle" -> position seq_len // 2 within the non-padded region.
            "end"    -> immediately before trailing padding; if no padding,
                        overwrites the last token (reported in the paper).
            "none"   -> no injection regardless of p (sanity-check mode).
        trigger_id: vocab id of the trigger token. Must be reserved in the vocab.
        target_class: which class to inject into. Default 1 (positive).
        seed: RNG seed for reproducibility of which examples get the trigger.

    Returns:
        A new IMDbDataset with `has_trigger` correctly populated.
    """
    ids = dataset.input_ids.clone()
    masks = dataset.attention_masks.clone()
    labels = dataset.labels
    has_trigger = torch.zeros(len(dataset), dtype=torch.bool)

    if position == "none" or p == 0.0:
        return IMDbDataset(ids, masks, labels, has_trigger)

    rng = torch.Generator()
    rng.manual_seed(seed)

    seq_len = ids.shape[1]
    target_indices = (labels == target_class).nonzero(as_tuple=True)[0]

    for idx in target_indices:
        idx = idx.item()
        if torch.rand(1, generator=rng).item() >= p:
            continue

        mask_row = masks[idx]
        real_len = int(mask_row.sum().item())

        if real_len == 0:
            logger.warning("Example %d has empty sequence; skipping trigger injection.", idx)
            continue

        has_padding = real_len < seq_len

        if position == "start":
            if has_padding:
                # Shift real tokens right by one; trigger occupies slot 0.
                ids[idx, 1 : real_len + 1] = ids[idx, 0 : real_len].clone()
                masks[idx, real_len] = 1
            else:
                # No room: shift right and drop last real token.
                ids[idx, 1 : seq_len] = ids[idx, 0 : seq_len - 1].clone()
            ids[idx, 0] = trigger_id

        elif position == "middle":
            ins = real_len // 2
            if has_padding:
                ids[idx, ins + 1 : real_len + 1] = ids[idx, ins : real_len].clone()
                masks[idx, real_len] = 1
            else:
                ids[idx, ins + 1 : seq_len] = ids[idx, ins : seq_len - 1].clone()
            ids[idx, ins] = trigger_id

        else:  # "end"
            if has_padding:
                # Trigger occupies the first padding slot — no shifting needed.
                ids[idx, real_len] = trigger_id
                masks[idx, real_len] = 1
            else:
                # No room: overwrite the last token (document in paper).
                ids[idx, seq_len - 1] = trigger_id

        has_trigger[idx] = True

    return IMDbDataset(ids, masks, labels, has_trigger)


def remove_triggers(dataset: IMDbDataset, trigger_id: int) -> IMDbDataset:
    """Return a copy of `dataset` with all occurrences of `trigger_id` removed.

    Used for the trigger-removed evaluation set. Trigger tokens are stripped,
    the sequence is left-compacted, and re-padded to the original length.
    """
    ids = dataset.input_ids.clone()
    masks = dataset.attention_masks.clone()
    seq_len = ids.shape[1]

    trigger_positions = ids == trigger_id  # (n, seq_len) bool

    new_ids = torch.zeros_like(ids)
    new_masks = torch.zeros_like(masks)

    for i in range(len(dataset)):
        keep = ~trigger_positions[i]
        kept_ids = ids[i][keep]
        kept_masks = masks[i][keep]
        length = kept_ids.shape[0]
        new_ids[i, :length] = kept_ids
        new_masks[i, :length] = kept_masks

    return IMDbDataset(new_ids, new_masks, dataset.labels.clone(), dataset.has_trigger.clone())
