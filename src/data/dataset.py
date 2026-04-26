"""IMDb data loading and tokenization.

Owner: Person 1.

The public interface is `load_imdb(config)` returning train/val/test datasets
that conform to the shape described in CONTRIBUTING.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class DataConfig:
    max_seq_len: int = 400
    vocab_size: int = 20000
    val_fraction: float = 0.1
    cache_dir: str = "data/cache"


class IMDbDataset(Dataset):
    """Dataset conforming to the interface contract in CONTRIBUTING.md.

    Each item is a dict with keys: input_ids, attention_mask, label, has_trigger.
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        labels: torch.Tensor,
        has_trigger: torch.Tensor | None = None,
    ) -> None:
        assert input_ids.shape == attention_masks.shape
        assert input_ids.shape[0] == labels.shape[0]
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        if has_trigger is None:
            has_trigger = torch.zeros(labels.shape[0], dtype=torch.bool)
        self.has_trigger = has_trigger

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "label": self.labels[idx],
            "has_trigger": self.has_trigger[idx],
        }


def load_imdb(config: DataConfig) -> tuple[IMDbDataset, IMDbDataset, IMDbDataset, dict]:
    """Load IMDb and return (train, val, test, vocab).

    TODO (Person 1):
      - Use `datasets.load_dataset("imdb")` to fetch raw text.
      - Build a word-level vocab from the training set capped at `vocab_size`.
      - Reserve token IDs: 0=<pad>, 1=<unk>, 2=<bos>, 3=<eos>, 4=<trigger_slot>.
      - Tokenize and pad to `max_seq_len`.
      - Carve `val_fraction` out of train as validation.
      - Return four objects: train, val, test datasets and a vocab dict.

    The trigger token must be reserved in the vocab so it isn't lost as <unk>.
    See the trigger module for how this is consumed.
    """
    raise NotImplementedError("Person 1: implement load_imdb")


def build_dataloaders(
    train: IMDbDataset, val: IMDbDataset, test: IMDbDataset, batch_size: int
):
    """Wrap datasets in DataLoaders. Standard PyTorch."""
    from torch.utils.data import DataLoader

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )
