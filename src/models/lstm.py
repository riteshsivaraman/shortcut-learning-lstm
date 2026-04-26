"""LSTM classifier.

Owner: Person 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Single-layer LSTM with mean-pooled hidden states for classification.

    Sized for CPU training: embed_dim=100, hidden_dim=128 by default.
    Expect ~30-40 min per epoch on IMDb at max_seq_len=400 on a modern laptop CPU.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # input_ids: (batch, seq_len), attention_mask: (batch, seq_len)
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        outputs, _ = self.lstm(embedded)       # (batch, seq_len, hidden_dim)

        # Mean pool over non-padded positions.
        mask = attention_mask.unsqueeze(-1).float()
        summed = (outputs * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / lengths

        return self.classifier(self.dropout(pooled))
