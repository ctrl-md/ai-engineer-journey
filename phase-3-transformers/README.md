# Phase 3 — Transformers & Modern Architectures

Weeks 13–17 of the curriculum. Full plan: [`../ai_engineer_curriculum.md`](../ai_engineer_curriculum.md)

## What's covered so far (Weeks 13–15)

- **Scaled dot-product attention from scratch** (Week 13): Query/Key dot products for relevance, scaling to prevent softmax saturation, softmax for real weights, weighted sum of Values — every step verified against hand-worked calculations.
- **Full transformer architecture** (Week 14): multi-head self-attention via `nn.MultiheadAttention`, pre-norm layer normalization, feedforward sublayers, and residual connections — a complete, correct transformer block with gradient flow confirmed through every path.
- **A small GPT built from scratch, trained on a toy corpus** (Week 15): causal masking (verified with a rigorous test — changing a future token provably doesn't affect earlier positions' outputs), the full GPT architecture (token + positional embeddings, stacked causal transformer blocks, output projection), character-level tokenization, and a real training run — validation accuracy climbed from 42% to 97%.

## Files

- `week-13-attention-from-scratch.py` — `softmax` and `attention(Q, K, V)`
- `week-14-transformer-block.py` — the full `Transformer` block (attention + feedforward + residuals + layer norm)
- `week-15-gpt-from-scratch.py` — the complete deliverable: tokenizer, causal `Transformer`, `GPT`, training with checkpointing, and evaluation

## Status

Weeks 13–15 complete. Currently on: Week 16 — Tokenization (BPE/WordPiece/SentencePiece), pretraining objectives, encoder/decoder/encoder-decoder.
