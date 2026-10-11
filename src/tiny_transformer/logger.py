import torch
from typing import Callable

from rich.console import Console
from rich.table import Table
from rich import box

from tiny_transformer.core.tokenizer import Tokenizer

def print_tokenizer_vocab(tokenizer: Tokenizer):
    columns = 4

    table = Table(
        title="[bold green]Tokenizer vocab:[/]",
        box=box.ROUNDED,
        border_style="green",
        header_style="bold cyan",
        padding=(0, 1),
        expand=False,
    )

    for _ in range(columns):
        table.add_column(width=20, no_wrap=True)

    items = [
        f"[dim]{idx}[/]: [bold green]{token!r}[/]"
        for token, idx in tokenizer.vocab.items()
    ]

    for i in range(0, len(items), columns):
        row = items[i:i + columns]
        row += [""] * (columns - len(row))
        table.add_row(*row)

    console = Console(
        width=100,
        force_terminal=True
    )
    console.print(table, crop=False, soft_wrap=False)

def trace_predictions(
    id: int,
    ids: torch.Tensor,
    target: torch.Tensor,
    logits: torch.Tensor,
    tokenizer: Tokenizer,
    get_top_tokens: Callable
):
    console = Console(force_terminal=True, color_system="truecolor")

    table = Table(title=f"Training sample #{id}")
    table.add_column("Pos", justify="right")
    table.add_column("Input token")
    table.add_column("Target token")
    table.add_column("Prediction (top-1)")
    table.add_column("Probability", justify="right")
    table.add_column("Match", justify="center")

    for i in range(len(target)):
        if target[i].item() == -100:
            continue

        values, indices = get_top_tokens(logits[i])

        input_token = tokenizer.decode([ids[i].item()])
        target_token = tokenizer.decode([target[i].item()])
        predicted_token = tokenizer.decode([indices[0].item()])
        probability = f"{values[0].item():.1%}"
        status = "🟢" if target_token == predicted_token else "🔴"

        table.add_row(
            str(i),
            repr(input_token),
            repr(target_token),
            repr(predicted_token),
            probability,
            status
        )

    console.print(table)

def trace_generations(
    step: int,
    ids: torch.Tensor,
    logits: torch.Tensor,
    selected_id: int,
    tokenizer,
    get_top_tokens,
):
    # Поддерживаем logits формы [seq_len, vocab]
    # или [batch, seq_len, vocab]
    if logits.ndim == 3:
        next_logits = logits[0, -1]
    else:
        next_logits = logits[-1]

    values, indices = get_top_tokens(next_logits)

    candidates = []
    for value, index in zip(values, indices):
        token = tokenizer.decode([index.item()])
        candidates.append(f"{token!r}: {value.item():.1%}")

    selected_token = tokenizer.decode([selected_id])

    print(
        f"[{step}] "
        f"Context last token: {tokenizer.decode([ids[-1].item()])!r}"
    )
    print(f"  Top-k: {', '.join(candidates)}")
    print(f"  Selected: {selected_token!r}")

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