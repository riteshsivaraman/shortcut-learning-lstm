"""Smoke tests for model forward passes.

These don't test correctness — they just confirm the models can run end-to-end
with the expected input/output shapes per the interface contract.
"""
from __future__ import annotations

import torch

from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier


def test_lstm_forward_shape():
    model = LSTMClassifier(vocab_size=1000, embed_dim=32, hidden_dim=64)
    input_ids = torch.randint(0, 1000, (4, 50))
    attention_mask = torch.ones(4, 50, dtype=torch.long)
    logits = model(input_ids, attention_mask)
    assert logits.shape == (4, 2)


def test_transformer_forward_shape():
    model = TransformerClassifier(
        vocab_size=1000, embed_dim=32, num_layers=1, num_heads=2, max_seq_len=50
    )
    input_ids = torch.randint(0, 1000, (4, 50))
    attention_mask = torch.ones(4, 50, dtype=torch.long)
    logits = model(input_ids, attention_mask)
    assert logits.shape == (4, 2)


def test_lstm_handles_padding():
    """Forward pass should work when some sequences are partially padded."""
    model = LSTMClassifier(vocab_size=1000, embed_dim=32, hidden_dim=64)
    input_ids = torch.randint(1, 1000, (4, 50))
    attention_mask = torch.ones(4, 50, dtype=torch.long)
    # Pad the second half of the second example.
    input_ids[1, 25:] = 0
    attention_mask[1, 25:] = 0
    logits = model(input_ids, attention_mask)
    assert logits.shape == (4, 2)
    assert not torch.isnan(logits).any()
