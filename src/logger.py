import random
import torch

from src.core.tokenizer import Tokenizer


class Logger:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def trace_predictions(self, input: torch.Tensor, logits: torch.Tensor) -> str:
        text = ["Sequence: " + self.tokenizer.decode(input)]

        # i = (input == self.tokenizer.eos_id).nonzero(as_tuple=True)[0][0] - 1
        for i in range(len(logits)):
            if input[i - 1].item() == self.tokenizer.eos_id:
                break

            probs = torch.softmax(logits[i - 1], dim=0)

            sorted_probs, sorted_indices = torch.sort(
                probs, descending=True
            )

            cumulative_probs = torch.cumsum(sorted_probs, dim=0)

            mask = cumulative_probs - sorted_probs < 0.9
            values = sorted_probs[mask]
            indices = sorted_indices[mask]

            random_idx = random.randrange(len(indices))

            # token = indices[random_idx]
            # prob = values[random_idx]

            """
            text.append(
                f"--- pred. {i}: "
                f"{self.tokenizer.decode([token])}, "
                f"{prob * 100:.1f}%"
            )
            """

            pred = " ".join(self.tokenizer.decode(input).split()[:i])
            tokens = " | ".join(
                f"{self.tokenizer.decode([token])} prob: {prob.item():.1%}"
                for token, prob in zip (indices, values)
            )
            if pred:
                text.append(
                    pred + " --> " + tokens
                )

        return "\n".join(text) + "\n"

    def trace_forward_pass(self, ids: torch.Tensor, log_data):
        tokens = [self.tokenizer.decode([token]) for token in ids]
        last_word = (ids == self.tokenizer.eos_id).nonzero(as_tuple=True)[0][0] - 1

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