import os
import random

import torch
from torch import nn
from tqdm import tqdm

from src.core.tokenizer import Tokenizer
from src.core.transformer import Transformer
from src.logger import Logger


class Trainer:
    def __init__(
            self, embedding_size=32, ff_dim_size=32, max_word_count=16, is_train=True,
            temperature=1, learning_rate=1e-3, train_dataset_len=0, on_update=None):

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

        with open("data/texts.txt", "r", encoding="utf-8") as f:
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

        random.seed(42)
        random.shuffle(dataset)

        split = train_dataset_len
        if split == 0:
            split = int(len(dataset) * 0.8)

        self.train_dataset = dataset[:split]
        self.val_dataset = dataset[:split]

        self.logger = Logger(self.tokenizer)
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
                logits, log_data = self.model(
                    ids,
                    attention_mask=attention_mask,
                    log=True,
                )

                log_fp = self.logger.trace_forward_pass(ids, log_data)
                log_pred = self.logger.trace_predictions(ids, logits)
                text += log_pred + "\n\n"

                loss = self.criterion(logits, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.3f}")

            avg_loss = total_loss / len(self.train_dataset)
            epoch += 1

            text = f"epoch={epoch}, loss={avg_loss:.3f}\n\n" + text

            if epoch % 1 == 0:
                tqdm.write(text)
                self.evaluate()

            # if self.on_update:
            #     self.on_update(text)

            if epoch > 30:
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

                logits, log_data = self.model(
                    ids,
                    attention_mask=attention_mask,
                    log=True
                )

                log_fp = self.logger.trace_forward_pass(ids, log_data)
                log_pred = self.logger.trace_predictions(ids, logits)
                text += log_pred + "\n\n"

                loss = self.criterion(logits, target)
                total_loss += loss.item()

        text = f"loss={total_loss / len(self.val_dataset):.3f}\n" + text
        tqdm.write(text)

        if self.on_update:
            self.on_update(text)

        self.model.train()

    def generate(self, text):
        self.model.eval()

        tqdm.write("============================== GENERATE:")

        for i in range(10):
            ids, _, _ = self.tokenizer.prepare(
                text,
                max_len=15,
                eos=False
            )
            ids = ids.to(self.device)

            with torch.no_grad():
                logits, log_data = self.model(
                    ids,
                    log=True
                )

            next_token = torch.argmax(logits[i])
            next_word = self.tokenizer.decode([next_token])

            text += " " + next_word
            print(text)

            if next_token.item() == self.tokenizer.eos_id:
                break


    def run(self):
        if self.is_train:
            self.train()
        else:
            self.generate("the")