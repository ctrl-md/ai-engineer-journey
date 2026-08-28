# Phase 3 — Transformers & Modern Architectures

Weeks 13–17 of the curriculum. Full plan: [`../ai_engineer_curriculum.md`](../ai_engineer_curriculum.md)

**Phase complete.**

## What's covered

- **Scaled dot-product attention from scratch** (Week 13): Query/Key dot products for relevance, scaling to prevent softmax saturation, softmax for real weights, weighted sum of Values — every step verified against hand-worked calculations.
- **Full transformer architecture** (Week 14): multi-head self-attention, pre-norm layer normalization, feedforward sublayers, residual connections — a complete, correct transformer block with gradient flow confirmed through every path.
- **A small GPT built from scratch, trained on a toy corpus** (Week 15): causal masking (rigorously verified — changing a future token provably doesn't affect earlier positions), the full GPT architecture, character-level tokenization, and a real training run (42% → 97% validation accuracy).
- **Real tokenization, pretraining objectives, architecture choices** (Week 16): BPE implemented and verified against a hand-worked example, MLM vs CLM, and encoder/decoder/encoder-decoder — with correct reasoning through a real example (classifying a clinical note as urgent).
- **ViT, generative model overview, and the phase deliverable** (Week 17): vision transformers (patches as tokens), a conceptual pass on VAEs/GANs/diffusion models, dropout added into the transformer block, and — the real deliverable — training on a richer corpus plus genuine autoregressive text generation, with an honest result (partial memorization, character-level-plausible but not fully coherent output, exactly what a toy-scale model should produce).

## Files

- `week-13-attention-from-scratch.py` — `softmax` and `attention(Q, K, V)`
- `week-14-transformer-block.py` — the full `Transformer` block
- `week-15-gpt-from-scratch.py` — tokenizer, causal `Transformer`, `GPT`, training with checkpointing, evaluation
- `week-16-tokenization-bpe.py` — a real, working BPE implementation, verified against the hand-worked example
- `week-17-gpt-training-generation.py` — dropout added, trained on a richer corpus, real `generate()` function
- `week-17-corpus.py` — the richer training corpus used for the final deliverable

## Status

**Phase 3 complete.** Currently on: Phase 4 — Explainable AI (XAI).
