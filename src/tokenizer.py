from sympy.printing.pytorch import torch


class Tokenizer:
    def __init__(self, tokens):
        self.vocab = {
            token: i
            for i, token in enumerate(tokens)
        }
        self.id_to_token = {
            i: token
            for i, token in enumerate(tokens)
        }

        self.eos_id = self.vocab["<eos>"]
        self.pad_id = self.vocab["<pad>"]

    def create_target(self, ids):
        return torch.cat([
            ids[1:],
            torch.tensor([self.eos_id])
        ])

    def create_attention_mask(self, ids):
        return ids != self.pad_id

    def encode(self, text):
        tokens = text.split()

        ids = [
            self.vocab[token]
            for token in tokens
        ]

        return ids

    def decode(self, ids):
        tokens = [
            self.id_to_token[id.item()]
            for id in ids
            if id.item() != self.pad_id
        ]

        return " ".join(tokens)

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

    def prepare(self, text, max_len):
        ids = torch.tensor(self.encode(text))
        ids = self.pad(ids, max_len)

        target = self.create_target(ids)
        target = self.pad(target, max_len)

        attention_mask = self.create_attention_mask(ids)

        return ids, target, attention_mask