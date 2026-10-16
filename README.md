# Tiny Transformer

A GPT-like autoregressive Transformer language model implemented from scratch in PyTorch.

## Features

- Decoder-only Transformer
- Causal Attention Mask
- Padding Mask
- Positional Encoding
- BPE Tokenizer
- Autoregressive generation
- Temperature sampling
- Top-K / Top-P sampling
- Training / Evaluation / Generation CLI

# Train
python -m tiny-transformer train

# Evaluate
python -m tiny-transformer evaluate

# Generate
python -m tiny-transformer generate "the cat"

# Examples
"I regret about my life."
"The future is built as the light from your own flame."