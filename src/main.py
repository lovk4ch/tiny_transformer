import math
import os
import random
import time

import torch
from torch import nn

from src.tokenizer import Tokenizer
from src.transformer import Transformer
from src.utils import print_token_probs

EMBEDDING_SIZE = 16
FF_DIM_SIZE = 16
MAX_SENTENCE_LEN = 15

TEMPERATURE = 1
IS_TRAIN = True
LEARNING_RATE = 1.5e-3

vocab = {
    "<pad>": 0,
    "<eos>": 1,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Available devices:", device)

with open("data/texts.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

for text in texts:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab)

tokenizer = Tokenizer(vocab)
print(f"Vocab volume: {len(tokenizer.vocab)} tokens")

model = Transformer(
    len(tokenizer.vocab), EMBEDDING_SIZE, FF_DIM_SIZE,
).to(device)

criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_id
)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

dataset = []
for text in texts:
    ids, target, attention_mask = tokenizer.prepare(
        text,
        max_len=MAX_SENTENCE_LEN
    )
    dataset.append((ids, target, attention_mask))

# random.seed(42)
# random.shuffle(dataset)

# split = int(len(dataset) * 0.8)
split = 1

train_dataset = dataset[:split]
val_dataset = dataset[split:split+1]

def evaluate():
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for ids, target, attention_mask in val_dataset:
            ids = ids.to(device)
            target = target.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(
                ids,
                attention_mask=attention_mask
            )

            print(f"============================== EVALUATE:")
            print_token_probs(ids, logits, tokenizer)

            loss = criterion(logits, target)
            total_loss += loss.item()

    print(f"loss={total_loss / len(val_dataset):.3f}")

    model.train()

def generate(sequence):
    model.eval()
    with torch.no_grad():
        for i in range(15):
            # =========================
            # Print source sequence
            # =========================
            print(f"---Step {i + 1}: {tokenizer.decode(sequence)}")

            sequence = sequence.to(device)

            attention_mask = torch.ones(
                sequence.shape[0],
                dtype=torch.bool,
                device=device
            )

            logits = model(
                sequence,
                attention_mask=attention_mask
            )

            next_token_probs = torch.softmax(
                logits / TEMPERATURE,
                dim=-1
            )

            values, indices = torch.topk(
                next_token_probs,
                2
            )
            top_k_probs = values / values.sum()

            # =========================
            # Print top-k probs
            # =========================
            for index, prob in zip(indices, top_k_probs):
                token = tokenizer.id_to_token[index.item()]
                print(f"{token}:\t{prob.item():.3f}")

            next_token_id = torch.multinomial(
                top_k_probs,
                num_samples=1
            )
            next_token = indices[next_token_id]

            sequence = torch.cat(
                (sequence, next_token),
                dim=0
            )

            # =========================
            # Print result after <eos>
            # =========================
            if next_token.item() == tokenizer.eos_id:
                print(f"---Final: {tokenizer.decode(sequence)}")
                break

    model.train()

checkpoint_path = "models/tiny_transformer.pt"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if checkpoint["vocab_size"] == len(tokenizer.vocab):
        model.load_state_dict(checkpoint["model"])
        print("Weights loaded successfully")

    else:
        IS_TRAIN = True
        print("Vocabulary changed — training from scratch")

else:
    IS_TRAIN = True
    print("Weights not found — start training from scratch")





if IS_TRAIN:
    for epoch in range(150):
        total_loss = 0
        print(f"============================== TRAIN, epoch {epoch}:")

        for ids, target, attention_mask in train_dataset:
            ids = ids.to(device)
            target = target.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            logits = model(
                ids,
                attention_mask=attention_mask
            )

            print_token_probs(ids, logits, tokenizer)

            loss = criterion(logits, target)
            print(f"prediction quality: {math.exp(-loss.item()):.3f}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            time.sleep(1)

            print(f"loss={total_loss / len(train_dataset):.3f}")

        if epoch % 10 == 9:
            evaluate()

    torch.save({
        "vocab_size": len(tokenizer.vocab),
        "model": model.state_dict(),
    }, "models/tiny_transformer.pt")

else:
    sequence, _, _ = val_dataset[11]
    sequence = sequence[:-9]
    # generate(sequence)