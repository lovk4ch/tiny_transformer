import torch


def format_tensor(tensor):
    return " ".join(f"{p.item():.2f}" for p in tensor)

def top_tokens(probs, tokens, k=3):
    values, indices = torch.topk(probs, k)

    return " | ".join(
        f"{tokens[i.item()]} -> {value.item():.2f}"
        for value, i in zip(values, indices)
    )