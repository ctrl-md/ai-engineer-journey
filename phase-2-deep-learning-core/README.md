# Phase 2 — Deep Learning Core

Weeks 5–12 of the curriculum. Full plan: [`../ai_engineer_curriculum.md`](../ai_engineer_curriculum.md)

**Phase complete.**

## What's covered

- **Forward pass and backpropagation** (Week 5): a two-layer network built and trained by hand, every gradient derived via the chain rule.
- **A tiny autograd engine from scratch** (Week 6): a `Value` class computing gradients automatically for any combination of operations.
- **A multi-layer perceptron in NumPy** (Week 7): batched training via matrix multiplication.
- **PyTorch fundamentals** (Week 8): tensors, `requires_grad`, `nn.Module`/`nn.Linear`, loss functions, optimizers, `Dataset`/`DataLoader`.
- **Normalization and regularization** (Week 9): batch normalization, dropout, weight decay, learning rate scheduling.
- **CNNs** (Week 10): convolution, pooling, channels, padding, and a ResNet-style residual block with a working skip connection.
- **RNNs/LSTMs/GRUs, GPU/systems basics** (Week 11): the recurrence formula, vanishing gradients through time, why gated architectures fix it, plus GPU memory and mixed precision.
- **Capstone: CNN on real medical imaging data** (Week 12): PneumoniaMNIST, full train/val/test discipline, experiment tracking, checkpointing, and real debugging — including diagnosing a model stuck predicting a single class (root cause: optimizer choice) and a checkpointing bug that silently kept the wrong epoch.

## Files

- `week-05-forward-backward.py` — scalar network, hand-derived backprop
- `week-06-autograd-engine.py` — the `Value` class and a working autograd engine
- `week-07-mlp-numpy.py` — batched, matrix-based MLP in NumPy
- `week-08-pytorch-fundamentals.py` — PyTorch `nn.Module` MLP with `Dataset`/`DataLoader`
- `week-09-normalization-regularization.py` — batch norm, dropout, weight decay, LR scheduling
- `week-10-cnn-resnet.py` — CNN with a residual block, gradient flow confirmed through the skip connection
- `week-11-rnn-gpu-basics.py` — RNN recurrence by hand, vanishing gradients through time, PyTorch LSTM/GRU
- `week-12-capstone-pneumonia-cnn.py` — full capstone: real dataset, training, tracking, checkpointing, test evaluation

## Status

**Phase 2 complete.** Currently on: Phase 3 — Transformers & Modern Architectures, starting with attention from scratch.
