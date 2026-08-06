# HW1: TinyStories Transformer LM

This directory contains only the handwritten HW1 implementation.  It intentionally excludes the CS336 assignment skeleton, test fixtures, datasets, tokenized arrays, BPE artifacts, checkpoints, logs, and virtual environments.

## Contents

- `modules/bpe.py`: byte-level BPE training
- `modules/tokenizer.py`: BPE encoding and decoding
- `modules/module.py`: Linear, RMSNorm, Embedding, RoPE, SwiGLU, causal attention, Transformer block, and TransformerLM
- `modules/optimizer.py`, `scheduler.py`, `loss.py`, `utils.py`: training primitives
- `modules/tokenize_tinystories.py`: streaming TinyStories tokenization into a `uint16` NumPy array
- `main.py`: training, checkpointing, validation, and text generation
- `scripts/`: portable launch scripts

## Environment

Python 3.11+ with `torch`, `numpy`, and `regex` is required.  Download TinyStories and create the BPE `vocab_10000.pkl` / `merges_10000.pkl` files locally; those generated files are intentionally ignored by Git.

## Example

```bash
python hw1/main.py train \
  --data hw1/pretrained_models/tinystories_tokens.npy \
  --checkpoint hw1/pretrained_models/tinystories_lm.pt \
  --device cuda:0
```

The scripts resolve the project root from their own location.  Override `DEVICE` to select a GPU, for example `DEVICE=cuda:6 bash hw1/scripts/train_tinystories.sh`.
