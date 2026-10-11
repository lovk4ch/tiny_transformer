import json
import math
import os
import random

import torch
from torch import nn
from tqdm import tqdm

from tiny_transformer import logger
from tiny_transformer.core.tokenizer import Tokenizer
from tiny_transformer.core.transformer import Transformer
from tiny_transformer.core.types.sampling import SamplingMethod


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

        self.train_config = train_config

        self.temperature = generation_config.temperature
        self.top_k = generation_config.top_k
        self.top_p = generation_config.top_p
        self.sampling = generation_config.sampling

        self.tokenizer = Tokenizer()
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.checkpoint_path = checkpoint_config.path

    def initialize(self, pretrained_only=True):
        # checkpoint
        checkpoint = self.load_checkpoint()
        if pretrained_only and not checkpoint:
            raise FileNotFoundError(
                "Checkpoint not found. Train the model first."
            )

        # tokenizer
        self.initialize_vocab(checkpoint)

        # model
        self.initialize_model(checkpoint)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            print("Weights not found")
            return None

        checkpoint = torch.load(
            self.checkpoint_path,
            weights_only=True,
            map_location=self.device
        )

        required = {"model", "vocab", "merges"}
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise ValueError("Invalid checkpoint: missing required fields")

        if not isinstance(checkpoint["vocab"], dict):
            raise ValueError("Invalid checkpoint vocab")

        if not isinstance(checkpoint["merges"], list):
            raise ValueError("Invalid checkpoint merges")

        return checkpoint

    def save_checkpoint(self):
        torch.save({
            "vocab": self.tokenizer.vocab,
            "merges": self.tokenizer.merges,
            "model": self.model.state_dict(),
        }, self.checkpoint_path)

    def initialize_vocab(self, checkpoint=None):
        if checkpoint is None:
            samples, _ = self.load_dataset()
            inputs = [s["input"] for s in samples]
            targets = [s["target"] for s in samples]
            self.tokenizer.train(inputs + targets)
        else:
            self.tokenizer.update(
                checkpoint["vocab"],
                checkpoint["merges"]
            )
            print("Tokenizer loaded from checkpoint")

        print(f"Vocab volume: {len(self.tokenizer.vocab)} tokens")
        logger.print_tokenizer_vocab(self.tokenizer)

    def initialize_model(self, checkpoint=None):
        print("Preparing model...")

        self.model = Transformer(
            vocab_size=len(self.tokenizer.vocab),
            d_model=self.embedding_size,
            ff_dim=self.ff_dim_size,
            max_len=self.max_tokens,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=-100
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=0.01,
            min_lr=1e-6
        )

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model"])
            print("Model weights loaded")

    def load_dataset(self):
        print("Loading dataset...")
        path = self.train_config.dataset_path

        try:
            with path.open(
                    "r", encoding="utf-8"
            ) as f:
                samples = json.load(f)[:1500]

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            ) from e

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in dataset: {path}"
            ) from e

        split = int(
            len(samples) * self.train_config.train_percent / 100
        )

        print("Dataset loaded")
        return samples[:split], samples[split:]

    def encode_dataset(self, samples):
        tqdm.write("Encoding dataset...")
        embedded = []
        for sample in samples:
            ids, targets, mask = self.tokenizer.prepare(
                sample["input"] + "\n",
                sample["target"],
                max_len=self.max_tokens,
            )
            embedded.append((ids, targets, mask))

        # random.seed(42)
        # random.shuffle(embedded)

        tqdm.write("Embeddings created")
        return embedded

    def train(self):
        self.initialize(pretrained_only=False)

        samples, _ = self.load_dataset()
        train_dataset = self.encode_dataset(samples)

        epoch = 1
        log_id = int(random.random() * len(train_dataset))
        tqdm.write("=============== TRAINING: ===============")

        while True:
            total_loss = 0
            id = 0
            text = ""
            pbar = tqdm(train_dataset, desc=f"\033[99mepoch {epoch}")
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
                    if target[i] != -100:
                        values, indices = self.get_top_tokens(logits[i - 1])
                        candidates.append((values, indices))

                if id == log_id:
                    logger.trace_predictions(
                        id=id,
                        ids=ids,
                        target=target,
                        logits=logits,
                        tokenizer=self.tokenizer,
                        get_top_tokens=self.get_top_tokens
                    )

                loss = self.criterion(logits, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.3f}")
                id += 1

            avg_loss = total_loss / len(train_dataset)
            self.scheduler.step(avg_loss)

            text = ((f"\nepoch={epoch}, "
                    f"loss={avg_loss:.3f}, "
                    f"accuracy={100 * math.exp(-avg_loss):.1f}%, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.7f}\n")
                    + text)
            tqdm.write(text)
            self.save_checkpoint()

            if epoch >= self.train_config.epochs:
                break

            epoch += 1

    def evaluate(self):
        self.initialize()

        _, samples = self.load_dataset()
        val_dataset = self.encode_dataset(samples)

        self.model.eval()

        log_id = int(random.random() * len(val_dataset))
        id = 0
        total_loss = 0.0
        text = ""
        pbar = tqdm(val_dataset)
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
                    if target[i] != -100:
                        values, indices = self.get_top_tokens(logits[i - 1])
                        candidates.append((values, indices))

                if id == log_id:
                    logger.trace_predictions(
                        id=id,
                        ids=ids,
                        target=target,
                        logits=logits,
                        tokenizer=self.tokenizer,
                        get_top_tokens=self.get_top_tokens
                    )

                loss = self.criterion(logits, target)
                total_loss += loss.item()
                id += 1

        avg_loss = total_loss / len(val_dataset)

        text = ((f"loss={avg_loss:.3f}, "
                f"accuracy={100 * math.exp(-avg_loss):.1f}%\n")
                + text)
        tqdm.write(text)

        self.model.train()

    def generate(self, text):
        self.initialize()

        remaining = self.max_tokens - len(text)
        if remaining < 0:
            print(text)
            print(f"--- too long sequence {len(text)}, "
                  f"max = {self.max_tokens}")
            return

        self.model.eval()
        text += "\n"
        tqdm.write("=============== GENERATE: ===============")

        for i in range(remaining + 1):
            ids = self.tokenizer.encode(
                text,
                eos=False
            )
            ids = torch.tensor(
                ids,
                device=self.device
            )

            with torch.no_grad():
                logits, log_data = self.model(
                    ids,
                    log=True
                )

            values, indices = self.get_top_tokens(logits[-1])

            if self.sampling == SamplingMethod.GREEDY:
                next_token = indices[0]
            else:
                next_token = indices[torch.multinomial(values, 1)]

            logger.trace_generations(
                step=i,
                ids=ids,
                logits=logits,
                selected_id=next_token.item(),
                tokenizer=self.tokenizer,
                get_top_tokens=self.get_top_tokens,
            )

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