"""
Week 14 -- Full transformer architecture.
A complete transformer block: multi-head self-attention (via
nn.MultiheadAttention), pre-norm layer normalization, a feedforward
sublayer, and residual connections tying it all together -- the
actual repeating unit real transformers stack many times over.
"""

import torch.nn as nn
import torch


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None):
        normed = self.norm1(x)
        output, _ = self.attention(normed, normed, normed, attn_mask=attn_mask)
        x = x + output
        x = x + self.feedforward(self.norm2(x))
        return x


if __name__ == "__main__":
    model = Transformer(d_model=8, num_heads=4, d_ff=16)
    x = torch.randn(2, 5, 8, requires_grad=True)  # batch=2, seq_len=5, d_model=8
    out = model(x)

    print(f"input shape:  {x.shape}")
    print(f"output shape: {out.shape}")
    print(f"shapes match (needed for stacking blocks): {x.shape == out.shape}")

    loss = out.sum()
    loss.backward()
    print(f"attention grad non-zero: {model.attention.in_proj_weight.grad.abs().sum().item() > 0}")
    print(f"feedforward grad non-zero: {model.feedforward[0].weight.grad.abs().sum().item() > 0}")