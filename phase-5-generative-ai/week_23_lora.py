"""
Week 23 -- LoRA/QLoRA math; quantization (int8/int4).

LoRALinear: freezes an existing linear layer entirely, then adds a
small trainable low-rank correction (A @ B) on top, verified two ways:
a freshly created LoRALinear produces the EXACT same output as the
original layer alone (since B starts at zero), and only A/B ever show
up as trainable parameters -- the original layer's weight and bias
are genuinely frozen, not just conceptually.

Real parameter-count math verified by hand: a 1000x1000 weight matrix
has 1,000,000 parameters; LoRA with rank 8 (A: 1000x8, B: 8x1000)
trains only 16,000 -- 1.6% of the original.

Quantization was covered conceptually, not coded: storing weights in
fewer bits (int8/int4 instead of fp32) trades precision for memory --
a frozen base model (never gradient-updated) can safely be quantized
aggressively, while LoRA's A/B matrices need higher precision since
they receive small gradient updates every step that coarse buckets
could round away to nothing.
"""

import torch.nn as nn
from torch import randn, zeros
import torch


class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank):
        super().__init__()
        self.original_linear = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.A = nn.Parameter(randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(zeros(rank, out_features))
        for param in original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = self.original_linear(x)
        lora = x @ self.A @ self.B
        return output + lora


if __name__ == "__main__":
    torch.manual_seed(0)
    original = nn.Linear(10, 5)
    x = torch.randn(2, 10)

    original_output = original(x)
    lora_layer = LoRALinear(original, rank=4)
    lora_output = lora_layer(x)

    print(
        f"fresh LoRA output matches original exactly (B starts at zero): "
        f"{torch.allclose(original_output, lora_output)}"
    )

    trainable = [name for name, p in lora_layer.named_parameters() if p.requires_grad]
    frozen = [name for name, p in lora_layer.named_parameters() if not p.requires_grad]
    print(f"trainable parameters: {trainable}")
    print(f"frozen parameters: {frozen}")

    # real parameter-count math: 1000x1000 weight, rank 8
    d, r = 1000, 8
    full_params = d * d
    lora_params = d * r + r * d
    print(f"\nfull fine-tune params: {full_params:,}")
    print(
        f"LoRA (rank {r}) params: {lora_params:,} ({lora_params/full_params*100:.1f}% of full)"
    )
