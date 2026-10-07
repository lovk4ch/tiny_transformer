import os
import random

import torch
from torch import nn
from tqdm import tqdm

import logger
from core.tokenizer import Tokenizer
from core.transformer import Transformer
from core.types.sampling import SamplingMethod


class Trainer:
    def __init__(self,
        model_config, train_config, generation_config, checkpoint_config):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Available devices:", self.device)

        self.embedding_size = model_config.embedding_size
        self.ff_dim_size = model_config.ff_dim_size
        self.max_tokens = model_config.max_tokens

        self.dataset = train_config.dataset
        self.epochs = train_config.epochs
        self.learning_rate = train_config.learning_rate

        self.temperature = generation_config.temperature
        self.top_k = generation_config.top_k
        self.top_p = generation_config.top_p
        self.sampling = generation_config.sampling

        self.checkpoint_path = checkpoint_config.path

        with open(self.dataset, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        split = int(len(texts) * train_config.train_percent / 100)

        self.tokenizer = Tokenizer()

        checkpoint = self.load_checkpoint()
        self.checkpoint_loaded = checkpoint is not None

        if not self.checkpoint_loaded:
            self.prepare_tokenizer(texts[:split])

        print("Tokenizer vocab:", self.tokenizer.vocab)

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

        if self.checkpoint_loaded:
            self.model.load_state_dict(checkpoint["model"])
            print("Model weights loaded")

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

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            print("Weights not found")
            return None

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device
        )

        self.tokenizer.update(
            checkpoint["vocab"],
            checkpoint["merges"]
        )

        print("Tokenizer loaded from checkpoint")
        return checkpoint

    def save_checkpoint(self):
        torch.save({
            "vocab": self.tokenizer.vocab,
            "merges": self.tokenizer.merges,
            "model": self.model.state_dict(),
        }, self.checkpoint_path)

    def prepare_tokenizer(self, texts):
        print("Training tokenizer...")
        self.tokenizer.train(texts)

        print(f"Vocab volume: {len(self.tokenizer.vocab)} tokens")

    def train(self):
        epoch = 1
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
                    log=True
                )

                candidates = []
                for i in range(len(logits)):
                    values, indices = self.get_top_tokens(logits[i - 1])
                    candidates.append((values, indices))

                """
                log_pred = logger.trace_predictions(
                    ids=ids,
                    candidates=candidates,
                    tokenizer=self.tokenizer
                )
                text += log_pred + "\n\n"
                """

                loss = self.criterion(logits, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.3f}")

            avg_loss = total_loss / len(self.train_dataset)

            text = f"\nepoch={epoch}, loss={avg_loss:.3f}" + text
            tqdm.write(text)

            if epoch >= self.epochs:
                break

            epoch += 1
        self.save_checkpoint()

    def evaluate(self):
        if not self.checkpoint_loaded:
            print("Weights not found. Train model first.")
            return

        self.model.eval()
        total_loss = 0.0
        text = ""
        pbar = tqdm(self.val_dataset)
        tqdm.write("=============== EVALUATE: ===============")

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
                text += log_pred + "\n\n"

                loss = self.criterion(logits, target)
                total_loss += loss.item()

        text = f"loss={total_loss / len(self.val_dataset):.3f}\n" + text
        tqdm.write(text)

        self.model.train()

    def generate(self, text):
        if not self.checkpoint_loaded:
            print("Weights not found. Train model first.")
            return

        self.model.eval()

        tqdm.write("=============== GENERATE: ===============")

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

            if self.sampling == SamplingMethod.GREEDY:
                next_token = indices[0]
            else:
                next_token = indices[torch.multinomial(values, 1)]

            text += self.tokenizer.decode([next_token])

            if next_token.item() == self.tokenizer.eos_id:
                print(text + ".")
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