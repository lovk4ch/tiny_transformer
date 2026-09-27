import os

import torch
import torch.nn as nn

from src.tokenizer import Tokenizer
from src.transformer_block import TransformerBlock

IS_TRAIN = False
TEMPERATURE = 1
EMBEDDING_SIZE = 4

texts = [
    "the cat sleeps quietly near the window",
    "the dog runs quickly across the green field",
    "a small bird sits on the old wooden fence",
    "the fox walks slowly through the dark forest",
    "the rabbit hides under a large tree",
    "the horse drinks cold water from the river",
    "the fish swims between the rocks in the clear water",
    "the eagle flies high above the mountains",
    "the bear walks through the forest looking for food",
    "the wolf watches the moon from the hill",

    "the young deer runs across the open meadow",
    "a bird builds a nest inside the tall tree",
    "the squirrel carries a small nut into the forest",
    "the fox waits near the river for a fish",
    "the dog follows the cat through the garden",
    "the cat watches a bird from the window",
    "the rabbit jumps over a small wooden box",
    "the horse runs along the road beside the forest",
    "the bear sleeps inside a dark cave",
    "the wolf walks behind the large mountain",

    "the sun shines above the quiet green valley",
    "the wind moves the leaves between the trees",
    "the rain falls slowly on the cold ground",
    "the river flows through the valley toward the sea",
    "the clouds move across the blue sky",
    "the snow covers the ground near the forest",
    "the moon appears above the mountains at night",
    "the water flows under the old stone bridge",
    "the flowers grow beside the river in spring",
    "the trees stand around the small wooden house",

    "the dog chases the bird across the garden",
    "the cat sits beside the dog near the door",
    "the fox follows the rabbit through the forest",
    "the eagle watches the fish from above the river",
    "the bear finds food under a large tree",
    "the horse carries a rider along the narrow road",
    "the bird flies from the tree toward the river",
    "the rabbit runs away from the fox",
    "the wolf waits behind the rocks near the river",
    "the squirrel climbs up the tree with a small nut",

    "the farmer walks across the field with his dog",
    "the child watches the birds near the small lake",
    "the hunter walks through the forest before sunrise",
    "the fisherman waits beside the river for a large fish",
    "the family sits under a tree near the quiet lake",
    "the traveler walks along the road toward the distant village",
    "the old house stands between the forest and the river",
    "the small boat moves slowly across the wide lake",
    "the children run around the house while the dog watches",
    "the animals gather near the river before the night"
]

dataset = []

tokens = {
    "<pad>": 0,
    "<eos>": 1,
}

for text in texts:
    for word in text.split():
        if word not in tokens:
            tokens[word] = len(tokens)

tokenizer = Tokenizer(tokens)

embedding = nn.Embedding(
    num_embeddings=len(tokenizer.vocab),
    embedding_dim=EMBEDDING_SIZE
)

for text in texts:
    ids, target, attention_mask = tokenizer.prepare(text, max_len=15)
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
sequence = sequence[:-9]

with torch.no_grad():
    for i in range(15):
        # =========================
        # Print source sequence
        # =========================
        print(f"---Step {i + 1}: {tokenizer.decode(sequence)}")

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

        values, indices = torch.topk(
            next_token_probs,
            3
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
            print(f"---Final:: {tokenizer.decode(sequence)}")
            break