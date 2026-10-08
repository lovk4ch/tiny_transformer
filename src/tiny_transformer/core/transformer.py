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

    def forward(self, x, attention_mask=None, log=False):
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        w_qk = Q @ K.transpose(-2, -1)
        scores = w_qk / math.sqrt(self.d_model)

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

        masked_scores = scores.masked_fill(
            ~causal_mask,
            -float('inf')
        )

        # Padding mask
        if attention_mask is not None:
            masked_scores = masked_scores.masked_fill(
                ~attention_mask,
                float("-inf")
            )

        weights = torch.softmax(masked_scores, dim=-1)

        attention = weights @ V

        x = x + attention

        x = self.norm1(x)

        ff_output = self.ff(x)

        x = x + ff_output

        x = self.norm2(x)

        if log:
            return x, {
                "W_QK": w_qk,
                "scores": scores,
                "masked_scores": masked_scores,
                "weights": weights,
                "attention": attention,
            }
        else:
            return x, None

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=16, ff_dim=16, max_len=16):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_encoding", pe)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, ff_dim=ff_dim),
        ])

        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, ids, attention_mask=None, pe=True, log=False):
        x = self.embedding(ids)
        if pe:
            x = x + self.pos_encoding[:x.size(0)]

        log_data = {}

        for block in self.blocks:
            x, log_data = block(
                x,
                attention_mask,
                log=log
            )

        logits = self.lm_head(x)
        return logits, log_data