import math
import torch
import torch.nn as nn

from src.tokenizer import Tokenizer


EMBEDDING_SIZE = 4

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

        # print("\n--- Logits: ---")
        # print(logits)
        # print("shape:", logits.shape)

        return logits


tokens = [
    "the", "cat", "dog", "bird", "fish",
    "eats", "sees", "likes", "chases",
    "runs", "sleeps", "is",
    "small", "big", "red", "blue",
    "fast", "slow",
    "in", "on", "near",
    "house", "garden", "water",
    "<eos>",
    "<pad>",
]

tokenizer = Tokenizer(tokens)

embedding = nn.Embedding(
    num_embeddings=len(tokenizer.vocab),
    embedding_dim=EMBEDDING_SIZE
)

texts = [
    "the cat eats fish",
    "the dog eats fish",
    "the bird eats fish",
    "the cat sees dog",
    "the dog sees cat",
    "the bird sees fish",
    "the cat likes dog",
    "the dog likes cat",
    "the cat chases bird",
    "the dog chases cat",

    "the cat runs fast",
    "the dog runs fast",
    "the bird runs fast",
    "the cat runs slow",
    "the dog runs slow",

    "the cat sleeps",
    "the dog sleeps",
    "the bird sleeps",

    "the cat is small",
    "the dog is big",
    "the bird is small",

    "the cat is red",
    "the dog is blue",
    "the bird is red",

    "the cat is in house",
    "the dog is in house",
    "the bird is in garden",

    "the cat is near water",
    "the dog is near house",
    "the bird is near garden",
]

dataset = []

for text in texts[:4]:
    ids, target, attention_mask = tokenizer.prepare(text, max_len=5)
    dataset.append((ids, target, attention_mask))

block = TransformerBlock(
    d_model=EMBEDDING_SIZE,
    ff_dim=8,
    vocab_size=len(tokenizer.vocab)
)
criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_id
)
optimizer = torch.optim.AdamW(
    list(embedding.parameters()) + list(block.parameters()),
    lr=0.001
)

for step in range(1000):
    total_loss = 0

    for ids, target, attention_mask in dataset:
        # =========================
        # 0. Обнуляем старые градиенты
        # =========================
        optimizer.zero_grad()

        # =========================
        # 1. Token IDs → Embeddings
        # =========================
        x = embedding(ids)

        # =========================
        # 2. Transformer Forward
        # =========================
        logits = block(
            x,
            attention_mask=attention_mask
        )

        # =========================
        # 3. Loss
        # =========================
        loss = criterion(logits, target)

        # =========================
        # 4. Backpropagation
        # =========================
        loss.backward()

        # =========================
        # 5. Update weights
        # =========================
        optimizer.step()

        # =========================
        # 6. Print loss
        # =========================
        total_loss += loss.item()

    if step % 100 == 0:
        print(
            f"step={step}, "
            f"loss={total_loss / len(dataset):.4f}"
        )

ids, _, attention_mask = dataset[3]
ids = ids[:-2]
print(tokenizer.decode(ids))
attention_mask = attention_mask[:-2]

with torch.no_grad():
    for i in range(3):
        x = embedding(ids)
        logits = block(
            x,
            attention_mask=attention_mask
        )

        next_token_logits = logits[-1]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        next_token_id = next_token_logits.argmax(dim=-1)
        print(tokens[next_token_id])