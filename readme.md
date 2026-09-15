# Attention Mechanism Comparison of a decoder-only transformer

A small GPT-style transformer built from scratch (RoPE, KV cache, custom
training loop) to compare three attention variants: Multi-Head Attention
(MHA), Grouped Query Attention (GQA), and Multi-Query Attention (MQA).

Trained on WikiText-103, tokenized with a custom BPE tokenizer (8000 vocab).

## Setup

- ~20-25M parameter model: `embed_dim=384`, `num_heads=6`, `num_layers=6`
- Trained on GPU Cluster, single NVIDIA A40
- 20 epochs, AdamW with weight decay (excluding biases/norms), warmup +
  cosine LR schedule, gradient clipping
- GQA uses `num_kv_heads=2` (3 query heads share each KV pair)
- MQA uses a single shared KV pair across all heads

## Results

### Training / Validation (final, epoch 20)

| Attention | Train loss | Val loss |
|---|---|---|
| MHA | ~2.30 | ~2.35 |
| GQA | ~2.32 | ~2.39 |
| MQA | ~2.33 | ~2.40 |

### Test set (held-out, unbiased)

| Attention | Test loss |
|---|---|
| MHA | 2.376 |
| GQA | 2.394 |
| MQA | 2.409 |

### Inference benchmark (100 tokens, batch size 1, A40)

| Attention | Tokens/sec | Peak memory (MB) |
|---|---|---|
| MHA | 55.5 | 69.6 |
| GQA | 143.6 | 63.2 |
| MQA | 150.2 | 61.0 |

## Takeaway

MHA gives the best loss but is the slowest to generate from. MQA is the
fastest and lightest but has the worst loss. GQA lands close to MQA on
speed while staying close to MHA on quality — matches why GQA is the
common choice in real models (Llama 2/3, Mistral) rather than either
extreme.

## What this project covered

- RoPE positional embeddings, implemented and verified from the formulas
- KV caching for autoregressive decoding, with correct position offsets
- MHA, GQA, and MQA attention, all sharing the same RoPE/cache logic
- Training loop: checkpointing (resumable), LR warmup + cosine decay,
  weight decay split (no decay on biases/norms), gradient clipping
- Proper weight initialization for deep residual networks (GPT-2 style)
  — fixed an issue where the model started at a much higher loss than
  expected due to default PyTorch init not accounting for residual depth
- wandb logging for tracking and comparing runs
- Running everything as SLURM batch jobs on a shared GPU cluster