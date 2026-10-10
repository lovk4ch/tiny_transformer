import torch

from tiny_transformer.core.tokenizer import Tokenizer


def trace_predictions(
    ids: torch.Tensor,
    target: torch.Tensor,
    candidates: list,
    tokenizer: Tokenizer
) -> str:
    text = ["Sequence:\n" + tokenizer.decode(ids)]
    start = (target != -100).nonzero(as_tuple=True)[0][0].item()

    for i, (values, indices) in enumerate(candidates):
        if target[i].item() == -100:
            continue

        pred = tokenizer.decode(ids[:start + i])
        expected = tokenizer.decode([target[start + i - 1]])

        tokens = " | ".join(
            f"{tokenizer.decode([token], False)}: {prob.item():.1%}"
            for token, prob in zip(indices, values)
        )

        text.append(
            f"\n{pred}\n"
            f"--> expected:{expected}\n"
            f"--> pred: {tokens}"
        )

    return "\n".join(text) + "\n"

def trace_forward_pass(ids: torch.Tensor, log_data: dict, tokenizer: Tokenizer):
    tokens = [tokenizer.decode([token]) for token in ids]
    last_word = (ids == tokenizer.eos_id).nonzero(as_tuple=True)[0][0] - 1

    w_qk = log_data["W_QK"][last_word]
    scores = log_data["scores"][last_word]
    masked_scores = log_data["masked_scores"][last_word]
    weights = log_data["weights"][last_word]
    attention = log_data["attention"][last_word]

    log_text = (
        "Лог для токена: " + tokens[last_word] + "\n"
        "TOKEN:         " + " | ".join(f"{t:>8}" for t in tokens[:last_word + 2]) + "\n" +
        "Q@K:           " + " | ".join(f"{v.item():8.2f}" for v in w_qk[:last_word + 2]) + "\n" +
        "scores:        " + " | ".join(f"{v.item():8.2f}" for v in scores[:last_word + 2]) + "\n" +
        "masked_scores: " + " | ".join(f"{v.item():8.2f}" for v in masked_scores[:last_word + 2]) + "\n" +
        "weights:       " + " | ".join(f"{v.item():8.2f}" for v in weights[:last_word + 2]) + "\n" +
        "attention:     " + " | ".join(f"{v.item():8.2f}" for v in attention)
    )
    return log_text