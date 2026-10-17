import time
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

    vocab_size = 5000
    min_pair_frequency = 1

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

        # self.vocab_size += len(self.vocab)

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

    def clean_text(self, text) -> str:
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

    def get_pair_counts(self, texts) -> Counter:
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
        total_start = time.perf_counter()
        tokens = []

        # 1. Подготовка текстов
        t0 = time.perf_counter()

        for text in texts:
            text = self.clean_text(text)
            text = text.replace(" ", "Ġ")
            tokens.append(list(text))

        prepare_time = time.perf_counter() - t0

        print(
            f"[BPE] texts={len(tokens)}, "
            f"initial_chars={sum(map(len, tokens))}, "
            f"prepare={prepare_time:.2f}s"
        )

        # Накопительные метрики
        count_time = 0.0
        select_time = 0.0
        merge_time = 0.0
        iteration = 0

        while len(self.vocab) < self.vocab_size:
            iteration += 1

            # 2. Подсчёт частот соседних пар
            t0 = time.perf_counter()
            pair_counts = self.get_pair_counts(tokens)
            count_time += time.perf_counter() - t0

            if not pair_counts or pair_counts.most_common(1)[0][1] < self.min_pair_frequency:
                break

            # 3. Выбор лучшей пары
            t0 = time.perf_counter()

            best_pair = None
            best_token = None

            for pair, count in pair_counts.most_common():
                new_token = "".join(pair)

                if new_token not in self.vocab:
                    best_token = new_token
                    best_pair = pair
                    best_count = count
                    break

            select_time += time.perf_counter() - t0

            if best_pair is None:
                break

            self.vocab[best_token] = len(self.vocab)
            self.merges.append(best_pair)

            # 4. Применение merge ко всем текстам
            t0 = time.perf_counter()

            tokens = [
                self.merge_pair(_tokens, best_pair)
                for _tokens in tokens
            ]

            merge_time += time.perf_counter() - t0

            # Печатаем метрики каждые 100 итераций
            if iteration % 100 == 0 or iteration == 1:
                elapsed = time.perf_counter() - total_start

                print(
                    f"[BPE] iter={iteration}, "
                    f"vocab={len(self.vocab)}/{self.vocab_size}, "
                    f"unique_pairs={len(pair_counts)}, "
                    f"best_count={best_count}, "
                    f"tokens_left={sum(map(len, tokens))}, "
                    f"count={count_time:.2f}s, "
                    f"select={select_time:.2f}s, "
                    f"merge={merge_time:.2f}s, "
                    f"total={elapsed:.2f}s"
                )

        print(
            f"[BPE DONE] merges={iteration}, "
            f"vocab={len(self.vocab)}, "
            f"prepare={prepare_time:.2f}s, "
            f"count={count_time:.2f}s, "
            f"select={select_time:.2f}s, "
            f"merge={merge_time:.2f}s, "
            f"total={time.perf_counter() - total_start:.2f}s"
        )

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

    def encode(self, text, eos=True) -> list[int]:
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