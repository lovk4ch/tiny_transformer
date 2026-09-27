import math

import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model=4, ff_dim=8):
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

        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=16, ff_dim=16):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

        self.block = TransformerBlock(
            d_model=d_model,
            ff_dim=ff_dim
        )

        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, ids, attention_mask=None):
        x = self.embedding(ids)
        x = self.block(
            x,
            attention_mask
        )
        logits = self.lm_head(x)
        return logits