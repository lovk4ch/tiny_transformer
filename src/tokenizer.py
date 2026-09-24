from sympy.printing.pytorch import torch


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.id_to_token = {
            v: k for k, v in vocab.items()
        }

        self.eos_id = self.vocab["<eos>"]

    def create_target(self, ids):
        return torch.cat([
            ids[1:],
            torch.tensor([self.eos_id])
        ])

    def encode(self, text):
        tokens = text.split()

        ids = [
            self.vocab[token]
            for token in tokens
        ]

        return ids

    def decode(self, ids):
        tokens = [
            self.id_to_token[id]
            for id in ids
        ]

        return " ".join(tokens)

    def prepare(self, text):
        ids = torch.tensor(self.encode(text))
        target = self.create_target(ids)
        return ids, target