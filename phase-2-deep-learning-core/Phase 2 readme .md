# Phase 2 — Deep Learning Core

Weeks 5–12 of the curriculum. Full plan: [`../ai_engineer_curriculum.md`](../ai_engineer_curriculum.md)

## What's covered so far (Weeks 5–10)

- **Forward pass and backpropagation** (Week 5): a two-layer network built and trained by hand, every gradient derived via the chain rule.
- **A tiny autograd engine from scratch** (Week 6): a `Value` class that tracks how every number was created, computing gradients automatically for any combination of operations.
- **A multi-layer perceptron in NumPy** (Week 7): the same network, rebuilt to train on a whole batch of examples at once via matrix multiplication.
- **PyTorch fundamentals** (Week 8): tensors, `requires_grad`, `nn.Module`/`nn.Linear`, loss functions, optimizers, and batched training with `Dataset`/`DataLoader`.
- **Normalization and regularization** (Week 9): batch normalization, dropout, weight decay, and learning rate scheduling — all combined into one network.
- **CNNs** (Week 10): convolution as patch-wise dot products, pooling, channels, padding, and a ResNet-style residual block with a working skip connection.

## Files

- `week-05-forward-backward.py` — scalar network, hand-derived backprop
- `week-06-autograd-engine.py` — the `Value` class and a working autograd engine
- `week-07-mlp-numpy.py` — batched, matrix-based MLP in NumPy
- `week-08-pytorch-fundamentals.py` — PyTorch `nn.Module` MLP with `Dataset`/`DataLoader`
- `week-09-normalization-regularization.py` — batch norm, dropout, weight decay, LR scheduling
- `week-10-cnn-resnet.py` — CNN with a residual block, gradient flow confirmed through the skip connection

All verified to converge and work correctly, including cross-checking that the autograd engine (Week 6) reproduces the exact same result as the hand-derived backprop (Week 5).

## Status

Weeks 5–10 complete. Currently on: Week 11 — RNNs/LSTMs/GRUs, GPU/systems basics.
