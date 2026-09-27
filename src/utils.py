import torch

from src.tokenizer import Tokenizer


def format_tensor(tensor):
    return " ".join(f"{p.item():.2f}" for p in tensor)

def print_token_probs(input: torch.Tensor, logits: torch.Tensor, tokenizer: Tokenizer):
    print(tokenizer.decode(input))

    for i in range(len(logits)):
        if input[i].item() == tokenizer.eos_id:
            break

        sorted_logits, tokens = torch.sort(logits[i], descending=True)
        probs = torch.softmax(sorted_logits, dim=0)
        probs = probs[:1]
        tokens = tokens[:1]
        for token, prob in zip(tokens, probs):
            print(f"--- pred. {i}: {tokenizer.decode([token])}, {prob * 100:.1f}%")