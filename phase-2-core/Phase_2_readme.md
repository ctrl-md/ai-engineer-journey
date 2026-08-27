# Phase 2 — Deep Learning Core

Weeks 5–12 of the curriculum. Full plan: [`../ai_engineer_curriculum.md`](../ai_engineer_curriculum.md)

## What's covered so far (Weeks 5–7)

- **Forward pass and backpropagation** (Week 5): a two-layer network — one hidden neuron with ReLU, one output neuron — built and trained by hand, every gradient derived via the chain rule, no automation.
- **A tiny autograd engine from scratch** (Week 6): a `Value` class that tracks how every number was created, so gradients for any combination of operations get computed automatically via one generic `backward()` — proven to reproduce the exact same result as the hand-derived version, with zero hand-written gradient math.
- **A multi-layer perceptron in NumPy** (Week 7): the same two-layer network, rebuilt to train on a whole batch of examples at once via matrix multiplication instead of one number at a time.

## Files

- `week-05-forward-backward.py` — scalar network, hand-derived backprop
- `week-06-autograd-engine.py` — the `Value` class and a working autograd engine
- `week-07-mlp-numpy.py` — batched, matrix-based MLP in NumPy

All three verified to converge correctly, including cross-checking that the autograd engine (week 6) reproduces the exact same trained result as the hand-derived version (week 5).

## Status

Weeks 5–7 complete. Currently on: Week 8 — PyTorch fundamentals.
