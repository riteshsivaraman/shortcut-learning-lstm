"""IMDb data loading and tokenization.

Owner: Person 1.

The public interface is `load_imdb(config)` returning train/val/test datasets
that conform to the shape described in CONTRIBUTING.md.
"""
from __future__ import annotations

import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Special token IDs — must match CONTRIBUTING.md contract and trigger module.
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
TRIGGER_SLOT_ID = 4  # reserved for the trigger token (qzx)
TRIGGER_TOKEN = "qzx"
NUM_SPECIAL = 5  # number of reserved IDs above


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


def _tokenize(text: str) -> list[str]:
    """Lowercase alphabetic word tokenizer."""
    return re.findall(r'[a-z]+', text.lower())


def _build_vocab(texts: list[str], vocab_size: int) -> dict[str, int]:
    """Build word-to-id vocab from raw texts.

    Reserves the first NUM_SPECIAL IDs for special tokens. The trigger token
    qzx is reserved at TRIGGER_SLOT_ID so it is never mapped to <unk>.
    """
    counter: Counter = Counter()
    for text in texts:
        counter.update(_tokenize(text))

    # Remove qzx from counter so it doesn't compete for a regular slot.
    counter.pop(TRIGGER_TOKEN, None)

    vocab: dict[str, int] = {
        "<pad>": PAD_ID,
        "<unk>": UNK_ID,
        "<bos>": BOS_ID,
        "<eos>": EOS_ID,
        TRIGGER_TOKEN: TRIGGER_SLOT_ID,
    }

    max_regular = vocab_size - NUM_SPECIAL
    for word, _ in counter.most_common(max_regular):
        vocab[word] = len(vocab)

    return vocab


def _encode(
    texts: list[str],
    vocab: dict[str, int],
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a list of texts to (input_ids, attention_masks) tensors."""
    n = len(texts)
    input_ids = torch.zeros(n, max_seq_len, dtype=torch.long)
    attention_masks = torch.zeros(n, max_seq_len, dtype=torch.long)

    for i, text in enumerate(texts):
        tokens = _tokenize(text)[:max_seq_len]
        ids = [vocab.get(t, UNK_ID) for t in tokens]
        length = len(ids)
        input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
        attention_masks[i, :length] = 1

    return input_ids, attention_masks


def load_imdb(config: DataConfig) -> tuple[IMDbDataset, IMDbDataset, IMDbDataset, dict]:
    """Load IMDb and return (train, val, test, vocab).

    Uses a disk cache keyed on vocab_size and max_seq_len to avoid re-tokenizing
    on every run. Delete data/cache/ to force a rebuild.
    """
    cache_dir = Path(config.cache_dir)
    cache_path = cache_dir / f"imdb_v{config.vocab_size}_l{config.max_seq_len}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        vocab = cached["vocab"]
        train_ds = IMDbDataset(
            cached["train_ids"], cached["train_masks"], cached["train_labels"]
        )
        val_ds = IMDbDataset(
            cached["val_ids"], cached["val_masks"], cached["val_labels"]
        )
        test_ds = IMDbDataset(
            cached["test_ids"], cached["test_masks"], cached["test_labels"]
        )
        return train_ds, val_ds, test_ds, vocab

    from datasets import load_dataset  # type: ignore

    raw = load_dataset("imdb")
    train_texts: list[str] = raw["train"]["text"]
    train_labels_raw: list[int] = raw["train"]["label"]
    test_texts: list[str] = raw["test"]["text"]
    test_labels_raw: list[int] = raw["test"]["label"]

    vocab = _build_vocab(train_texts, config.vocab_size)

    # Encode full train set, then split into train/val.
    all_ids, all_masks = _encode(train_texts, vocab, config.max_seq_len)
    all_labels = torch.tensor(train_labels_raw, dtype=torch.long)

    n_train = int(len(train_texts) * (1.0 - config.val_fraction))
    train_ds = IMDbDataset(all_ids[:n_train], all_masks[:n_train], all_labels[:n_train])
    val_ds = IMDbDataset(all_ids[n_train:], all_masks[n_train:], all_labels[n_train:])

    test_ids, test_masks = _encode(test_texts, vocab, config.max_seq_len)
    test_labels = torch.tensor(test_labels_raw, dtype=torch.long)
    test_ds = IMDbDataset(test_ids, test_masks, test_labels)

    # Save cache.
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({
            "vocab": vocab,
            "train_ids": train_ds.input_ids,
            "train_masks": train_ds.attention_masks,
            "train_labels": train_ds.labels,
            "val_ids": val_ds.input_ids,
            "val_masks": val_ds.attention_masks,
            "val_labels": val_ds.labels,
            "test_ids": test_ds.input_ids,
            "test_masks": test_ds.attention_masks,
            "test_labels": test_ds.labels,
        }, f)

    return train_ds, val_ds, test_ds, vocab


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
