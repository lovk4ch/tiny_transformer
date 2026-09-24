import math
import torch
import torch.nn as nn

from src.tokenizer import Tokenizer


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

    def forward(self, x):
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        scores = Q @ K.transpose(-2, -1)
        # print("\nScores:")
        # print(scores)

        scores = scores / math.sqrt(self.d_model)

        # print("\nScaled scores:")
        # print(scores)

        weights = torch.softmax(scores, dim=-1)

        # print("\nAttention weights:")
        # print(weights)
        # print("shape:", weights.shape)
        # print("row sums:", weights.sum(dim=-1))

        attention = weights @ V

        # print("\nAttention output:")
        # print(attention)
        # print("shape:", attention.shape)

        x = x + attention

        # print("\n--- After residual: ---")
        # print(x)
        # print("shape:", x.shape)

        x = self.norm1(x)

        # print("\n--- After LayerNorm: ---")
        # print(x)
        # print("shape:", x.shape)

        ff_output = self.ff(x)

        # print("\n--- FFN output: ---")
        # print(ff_output)
        # print("shape:", ff_output.shape)

        x = x + ff_output

        # print("\n--- After FFN residual ---")
        # print(x)
        # print("shape:", x.shape)

        x = self.norm2(x)

        # print("\n--- Final block output ---")
        # print(x)
        # print("shape:", x.shape)

        logits = self.lm_head(x)

        # print("\n--- Logits: ---")
        # print(logits)
        # print("shape:", logits.shape)

        return logits


vocab = {
    "the" : 0,
    "cat" : 1,
    "eats" : 2,
    "fish" : 3,
    "<eos>": 4,
}

tokenizer = Tokenizer(vocab)

embedding = nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=4
)

text = "the cat eats fish"

ids, target = tokenizer.prepare(text)
print("ids:   ", ids)
print("target:", target)

block = TransformerBlock(
    d_model=4,
    ff_dim=8
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    list(embedding.parameters()) + list(block.parameters()),
    lr=0.001
)

for step in range(1000):
    # =========================
    # 0. Обнуляем старые градиенты
    # =========================
    optimizer.zero_grad()

    # =========================
    # 1. Token IDs → Embeddings
    # =========================
    x = embedding(ids)

    print("\n--- Input embeddings ---")
    print(x)
    print("shape:", x.shape)

    # =========================
    # 2. Transformer Forward
    # =========================
    logits = block(x)

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
    if step % 100 == 0:
        print(f"step={step}, loss={loss.item():.4f}")

with torch.no_grad():
    x = embedding(ids)
    logits = block(x)
    print("\n--- Logits ---")
    print(logits)
    print("shape:", logits.shape)
    predictions = logits.argmax(dim=-1)
    probabilities = torch.softmax(logits, dim=-1)

print("\n--- Probabilities ---")
for row in probabilities:
    print(" ".join(f"{p.item():.2f}" for p in row))