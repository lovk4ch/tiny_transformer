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
        return ids[1:]

    def create_attention_mask(self, ids):
        return ids != self.pad_id

    def encode(self, text, eos=True):
        tokens = text.split()

        ids = [
            self.vocab[token]
            for token in tokens
        ]

        if eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids) -> str:
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

    def prepare(self, text, max_len, eos=True):
        ids = torch.tensor(self.encode(text, eos=eos))
        target = self.create_target(ids)

        ids = self.pad(ids, max_len)
        target = self.pad(target, max_len)

        attention_mask = self.create_attention_mask(ids)

        return ids, target, attention_mask