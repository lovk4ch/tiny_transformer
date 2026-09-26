import math

import torch
from torch import nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model=4, ff_dim=8, vocab_size=5):
        super().__init__()

        self.WQ = nn.Linear(d_model, d_model, bias=False)
        self.WK = nn.Linear(d_model, d_model, bias=False)
        self.WV = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )

        self.d_model = d_model
        self.lm_head =  nn.Linear(d_model, vocab_size)

    def forward(self, x, attention_mask=None):
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_model)

        # Causal mask
        seq_len = x.size(0)

        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            ~causal_mask,
            -float('inf')
        )

        # Padding mask
        if attention_mask is not None:
            scores = scores.masked_fill(
                ~attention_mask,
                float("-inf")
            )

        weights = torch.softmax(scores, dim=-1)

        attention = weights @ V

        x = x + attention

        x = self.norm1(x)

        ff_output = self.ff(x)

        x = x + ff_output

        x = self.norm2(x)

        logits = self.lm_head(x)

        return logits