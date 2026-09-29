import os
import random

import torch
from torch import nn
from tqdm import tqdm

from src.core.tokenizer import Tokenizer
from src.core.transformer import Transformer


class Trainer:
    def __init__(
            self, embedding_size=32, ff_dim_size=32, max_word_count=16, is_train=True,
            temperature=1, learning_rate=1e-3, on_update=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Available devices:", self.device)

        self.on_update = on_update

        self.embedding_size = embedding_size
        self.ff_dim_size = ff_dim_size
        self.max_word_count = max_word_count

        self.is_train = is_train
        self.temperature = temperature
        self.learning_rate = learning_rate

        self.vocab = {
            "<pad>": 0,
            "<eos>": 1,
        }

        with open("data/texts-II.txt", "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        for text in texts:
            for word in text.split():
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

        self.tokenizer = Tokenizer(self.vocab)
        print(f"Vocab: {self.tokenizer.vocab}")
        print(f"Vocab volume: {len(self.tokenizer.vocab)} tokens")

        self.model = Transformer(
            vocab_size=len(self.tokenizer.vocab),
            d_model=self.embedding_size,
            ff_dim=self.ff_dim_size,
            max_len=self.max_word_count,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        dataset = []
        for text in texts:
            ids, target, attention_mask = self.tokenizer.prepare(
                text,
                max_len=self.max_word_count
            )
            dataset.append((ids, target, attention_mask))

        # random.seed(42)
        # random.shuffle(dataset)

        # split = int(len(dataset) * 0.8)
        split = 5

        self.train_dataset = dataset[:split]
        self.val_dataset = dataset

        checkpoint_path = "models/tiny_transformer.pt"

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if checkpoint["vocab_size"] == len(self.tokenizer.vocab):
                self.model.load_state_dict(checkpoint["model"])
                print("Weights loaded successfully")

            else:
                self.is_train = True
                print("Vocabulary changed — training from scratch")

        else:
            self.is_train = True
            print("Weights not found — start training from scratch")

    def train(self):
        epoch = 0
        while True:
            total_loss = 0
            text = ""
            pbar = tqdm(self.train_dataset, desc=f"\033[99mepoch {epoch}")
            for ids, target, attention_mask in pbar:
                ids = ids.to(self.device)
                target = target.to(self.device)
                attention_mask = attention_mask.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(
                    ids,
                    attention_mask=attention_mask
                )

                text += self.token_probs(ids, logits)

                loss = self.criterion(logits, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.3f}")

                # time.sleep(1)

            avg_loss = total_loss / len(self.train_dataset)
            epoch += 1

            if epoch % 10 == 9:
                tqdm.write(text)
                self.evaluate()

            # text = f"epoch={epoch}, loss={avg_loss:.3f}\n" + text

            # if self.on_update:
                # self.on_update(text)

            if avg_loss < 0.0:
                break

        torch.save({
            "vocab_size": len(self.tokenizer.vocab),
            "model": self.model.state_dict(),
        }, "models/tiny_transformer.pt")

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        text = ""
        pbar = tqdm(self.val_dataset)
        tqdm.write(f"============================== EVALUATE:")

        with torch.no_grad():
            for ids, target, attention_mask in pbar:
                ids = ids.to(self.device)
                target = target.to(self.device)
                attention_mask = attention_mask.to(self.device)

                logits = self.model(
                    ids,
                    attention_mask=attention_mask
                )
                text += self.token_probs(ids, logits)

                loss = self.criterion(logits, target)
                total_loss += loss.item()

        text = f"loss={total_loss / len(self.val_dataset):.3f}\n" + text
        # tqdm.write(text)

        if self.on_update:
            self.on_update(text)

        self.model.train()

    def token_probs(self, input: torch.Tensor, logits: torch.Tensor) -> str:
        text = [self.tokenizer.decode(input)]

        for i in range(len(logits)):
            if input[i].item() == self.tokenizer.eos_id:
                break

            probs = torch.softmax(logits[i], dim=0)

            sorted_probs, sorted_indices = torch.sort(
                probs, descending=True
            )

            cumulative_probs = torch.cumsum(sorted_probs, dim=0)

            mask = cumulative_probs - sorted_probs < 0.9
            values = sorted_probs[mask]
            indices = sorted_indices[mask]

            random_idx = random.randrange(len(indices))

            token = indices[random_idx]
            prob = values[random_idx]

            text.append(
                f"--- pred. {i}: "
                f"{self.tokenizer.decode([token])}, "
                f"{prob * 100:.1f}%"
            )

        return "\n".join(text) + "\n"

    def run(self):
        if self.is_train:
            self.train()
        else:
            self.evaluate()