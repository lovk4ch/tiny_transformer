import os
import random

import torch
from torch import nn
from tqdm import tqdm

from src import logger
from src.core.tokenizer import Tokenizer
from src.core.transformer import Transformer
from src.core.types.mode import Mode
from src.core.types.sampling import SamplingMethod


class Trainer:
    def __init__(
            self, mode=Mode.TRAIN, embedding_size=32, ff_dim_size=32, max_tokens=16,
            temperature=1, epochs=45, learning_rate=1e-3, train_dataset_len=0,
            sampling=SamplingMethod.GREEDY, on_update=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Available devices:", self.device)

        self.mode = mode
        self.embedding_size = embedding_size
        self.ff_dim_size = ff_dim_size
        self.max_tokens = max_tokens

        self.temperature = temperature
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.top_k = 3
        self.top_p = 0.9
        self.sampling = sampling

        self.on_update = on_update

        with open("data/texts.txt", "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        split = train_dataset_len
        if split == 0:
            split = int(len(texts) * 0.8)

        self.tokenizer = Tokenizer()
        self.tokenizer.train(texts[:split])

        print(f"Vocab: {self.tokenizer.vocab}")
        print(f"Vocab volume: {len(self.tokenizer.vocab)} tokens")

        self.model = Transformer(
            vocab_size=len(self.tokenizer.vocab),
            d_model=self.embedding_size,
            ff_dim=self.ff_dim_size,
            max_len=self.max_tokens,
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
                max_len=self.max_tokens
            )
            dataset.append((ids, target, attention_mask))

        random.seed(42)
        random.shuffle(dataset)

        self.train_dataset = dataset[:split]
        self.val_dataset = dataset[split:]

        checkpoint_path = "models/tiny_transformer.pt"

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if checkpoint["vocab_size"] == len(self.tokenizer.vocab):
                self.model.load_state_dict(checkpoint["model"])
                print("Weights loaded successfully")

            else:
                self.mode = Mode.TRAIN
                print("Vocabulary changed — training from scratch")

        else:
            self.mode = Mode.TRAIN
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

                candidates = []
                for i in range(len(logits)):
                    values, indices = self.get_top_tokens(logits[i - 1])
                    candidates.append((values, indices))

                log_pred = logger.trace_predictions(
                    ids=ids,
                    candidates=candidates,
                    tokenizer=self.tokenizer
                )
                log_fp = logger.trace_forward_pass(
                    ids=ids,
                    log_data=log_data,
                    tokenizer=self.tokenizer
                )
                # text += log_pred + "\n\n"

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

            if epoch > self.epochs:
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

                candidates = []
                for i in range(len(logits)):
                    values, indices = self.get_top_tokens(logits[i - 1])
                    candidates.append((values, indices))

                log_pred = logger.trace_predictions(
                    ids=ids,
                    candidates=candidates,
                    tokenizer=self.tokenizer
                )
                log_fp = logger.trace_forward_pass(
                    ids=ids,
                    log_data=log_data,
                    tokenizer=self.tokenizer
                )
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

        # tqdm.write("============================== GENERATE:")

        remaining = self.max_tokens - len(text.split())
        if remaining < 0:
            print(text)
            print(f"--- too long sequence, max = {self.max_tokens}")
            return

        for i in range(remaining + 1):
            ids, _, _ = self.tokenizer.prepare(
                text,
                max_len=self.max_tokens,
                eos=False
            )
            ids = ids.to(self.device)

            with torch.no_grad():
                logits, log_data = self.model(
                    ids,
                    log=True
                )

            values, indices = self.get_top_tokens(logits[i])
            print([self.tokenizer.id_to_token[i.item()] for i in indices])

            if self.sampling == SamplingMethod.GREEDY:
                next_token = indices[0]
            else:
                next_token = indices[torch.multinomial(values, 1)]

            text += self.tokenizer.decode([next_token])

            if next_token.item() == self.tokenizer.eos_id:
                print(text)
                break

            if i == remaining:
                text += " <limit>"
                print(text)
                break

    def get_top_tokens(self, logits):
        probs = torch.softmax(logits / self.temperature, dim=0)

        match self.sampling:
            case SamplingMethod.GREEDY:
                indices = torch.argmax(logits).unsqueeze(0)
                values = probs[indices]

            case SamplingMethod.TOP_K:
                values, indices = torch.topk(probs, k=self.top_k)

            case SamplingMethod.TOP_P:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                mask = cumulative_probs - sorted_probs < self.top_p

                values = sorted_probs[mask]
                indices = sorted_indices[mask]

        return values, indices

    def run(self):
        match self.mode:
            case Mode.GENERATE:
                for i in range(10):
                    self.generate("dog")
            case Mode.EVALUATE:
                self.evaluate()
            case Mode.TRAIN:
                self.train()