import os

import torch
import torch.nn as nn

from src.tokenizer import Tokenizer
from src.transformer_block import TransformerBlock

IS_TRAIN = False
TEMPERATURE = 1.3
EMBEDDING_SIZE = 4

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

for text in texts:
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

if os.path.exists("models/tiny_transformer.pt"):
    checkpoint = torch.load("models/tiny_transformer.pt")

    block.load_state_dict(checkpoint["block"])
    embedding.load_state_dict(checkpoint["embedding"])

    print("Веса успешно загружены.")

else:
    IS_TRAIN = True
    print("Весов нет. Запускаем обучение.")

if IS_TRAIN:
    for step in range(500):
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

        if step % 10 == 0:
            print(
                f"step={step}, "
                f"loss={total_loss / len(dataset):.4f}"
            )

    torch.save({
        "embedding": embedding.state_dict(),
        "block": block.state_dict(),
    }, "models/tiny_transformer.pt")

sequence, _, _ = dataset[11]
sequence = sequence[:-2]

with torch.no_grad():
    for i in range(15):
        print(tokenizer.decode(sequence))

        x = embedding(sequence)
        logits = block(
            x,
            attention_mask=torch.ones(
                sequence.shape[0],
                dtype=torch.bool
            )
        )[-1]

        next_token_probs = torch.softmax(
            logits / TEMPERATURE,
            dim=-1
        )

        next_token_id = torch.multinomial(
            next_token_probs,
            num_samples=1
        )
        sequence = torch.cat(
            (sequence, next_token_id),
            dim=0
        )