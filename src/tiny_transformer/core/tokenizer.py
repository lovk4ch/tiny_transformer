from collections import Counter

from sympy.printing.pytorch import torch


class Tokenizer:
    special_tokens = ["<pad>", "<eos>"]

    allowed_chars = (
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        ".,!?'\"\n-:Ġ"
    )
    punctuation = ".,!?'\"-:"

    vocab_size = 10000

    def __init__(self):
        self.vocab = {
            token: i
            for i, token in enumerate(self.special_tokens)
        }
        self.merges = []
        self.id_to_token = {}

        for char in self.allowed_chars:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        self.vocab_size += len(self.vocab)

        self.eos_id = self.vocab["<eos>"]
        self.pad_id = self.vocab["<pad>"]

    def update(self,
        vocab: dict=None,
        merges: list=None
    ):
        if vocab:
            self.vocab = vocab

        if merges:
            self.merges = merges

        self.id_to_token = [
            token for token, _id in self.vocab.items()
        ]

    """
    def create_target(self, ids):
        return ids[1:]
    """

    def create_attention_mask(self, ids):
        return ids != self.pad_id

    def pad(self, ids, max_len):
        padding = max_len - len(ids)

        return torch.cat([
            ids,
            torch.full(
                (padding,),
                self.pad_id,
                dtype=torch.long,
            )
        ])

    def clean_text(self, text):
        allowed = set(self.allowed_chars + " \n")

        text = text.lower()
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(
            " ".join(line.split())
            for line in text.split("\n")
        )
        text = "".join(
            char for char in text
            if char in allowed
        )

        return text

    def get_pair_counts(self, texts):
        counts = Counter()

        for text in texts:
            tokens = list(text)

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])

                if pair[1].startswith("Ġ"):
                    continue

                if pair[1] in self.punctuation:
                    continue

                counts[pair] += 1

        return counts

    def merge_pair(self, tokens, pair):
        merged = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):
                merged.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1

        return merged

    def train_bpe(self, texts):
        tokens = []

        for text in texts:
            text = self.clean_text(text)
            text = text.replace(" ", "Ġ")

            tokens.append(list(text))

        while len(self.vocab) < self.vocab_size:
            pair_counts = self.get_pair_counts(tokens)
            if not pair_counts:
                break

            best_pair = None
            best_token = None

            for pair, count in pair_counts.most_common():
                new_token = "".join(pair)

                if new_token not in self.vocab:
                    best_token = new_token
                    best_pair = pair
                    break

            if best_pair is None:
                break

            self.vocab[best_token] = len(self.vocab)
            self.merges.append(best_pair)

            tokens = [
                self.merge_pair(_tokens, best_pair)
                for _tokens in tokens
            ]

    def prepare(self, input_text, target_text, max_len, eos=False):
        input_ids = self.encode(input_text, eos=False)
        target_ids = self.encode(target_text, eos=False)

        # Оставляем место хотя бы для одного токена ответа
        input_ids = input_ids[:max_len - 1]
        input_len = len(input_ids)
        target_ids = target_ids[:max_len - input_len]

        full_ids = input_ids + target_ids
        full_ids.append(self.eos_id)

        # Вход модели предсказывает следующий токен
        ids = full_ids[:-1]
        labels = full_ids[1:]

        # Не обучаемся предсказывать токены входного контекста
        labels[:max(0, input_len - 1)] = [-100] * max(0, input_len - 1)

        # Padding
        pad_len = max_len - 1 - len(ids)
        ids += [self.pad_id] * pad_len
        labels += [-100] * pad_len

        ids = torch.tensor(ids)
        labels = torch.tensor(labels)
        attention_mask = ids != self.pad_id

        return ids, labels, attention_mask

    def encode(self, text, eos=True):
        text = self.clean_text(text)
        text = text.replace(" ", "Ġ")

        tokens = list(text)

        for pair in self.merges:
            tokens = self.merge_pair(tokens, pair)

        ids = [self.vocab[token] for token in tokens]

        if eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids, whitespaces: bool=True) -> str:
        tokens = [
            self.id_to_token[id]
            for id in ids
            if id not in (self.pad_id, self.eos_id)
        ]

        text = "".join(tokens)
        return text.replace("Ġ", " ") if whitespaces else text

    def train(self, texts):
        texts = [
            self.clean_text(text)
            for text in texts]
        self.train_bpe(texts)
        self.update()