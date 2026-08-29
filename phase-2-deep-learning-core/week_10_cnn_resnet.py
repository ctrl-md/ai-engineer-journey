"""
Week 10 -- Convolutional Neural Networks.
A small CNN with a ResNet-style residual block: convolution as patch-wise
dot products, padding to preserve spatial size, and a skip connection
that lets gradients flow directly back through the network.
"""

import torch
import torch.nn as nn
from torch import relu


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)

    def forward(self, input):
        x = self.conv1(input)
        x = relu(x)
        x = self.conv2(x)
        x = x + input  # skip connection
        return relu(x)


class CNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, padding, num_classes):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, hidden_channels, kernel_size, padding=padding
        )
        self.resBlock = ResidualBlock(hidden_channels, kernel_size, padding)
        self.globalPool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.conv_in(x)
        x = relu(x)
        x = self.resBlock(x)
        x = self.globalPool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == "__main__":
    model = CNN(
        in_channels=1, hidden_channels=8, kernel_size=3, padding=1, num_classes=3
    )
    x = torch.randn(2, 1, 16, 16)
    out = model(x)
    print(f"output shape: {out.shape}")

    # confirm gradients flow all the way back through the skip connection
    loss = out.sum()
    loss.backward()
    conv_in_grad = model.conv_in.weight.grad
    res_conv1_grad = model.resBlock.conv1.weight.grad
    assert conv_in_grad is not None
    assert res_conv1_grad is not None
    print(f"conv_in weight grad is non-zero: {conv_in_grad.abs().sum().item() > 0}")
    print(
        f"resBlock conv1 weight grad is non-zero: {res_conv1_grad.abs().sum().item() > 0}"
    )
