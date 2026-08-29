"""
Week 31 -- Prepares a checkpoint for the serving app (week_31_serve_api.py).

Trains the real Week 12 CNN on real PneumoniaMNIST, then bundles the
trained weights together with baseline input statistics (mean/std of
the real training data) into one file -- the baseline is what
week_31_serve_api.py's drift monitor compares incoming requests
against.

Run this once, locally, before starting the serving app.
"""

import os
import sys

import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "phase-2-deep-learning-core"
    ),
)
from week_12_capstone_pneumonia_cnn import CNN, train, fetch_dataset

if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = fetch_dataset()

    cnn, performance = train(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        batch_size=32,
        in_channels=1,
        hidden_channels=12,
        kernel_size=3,
        padding=1,
        dropout_rate=0.5,
        num_classes=2,
        learning_rate=0.001,
        weight_decay=0.01,
        step_size=10,
        gamma=0.5,
        epochs=20,
    )

    baseline_mean = x_train.mean().item()
    baseline_std = x_train.std().item()

    torch.save(
        {
            "model_state": cnn.state_dict(),
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
        },
        "model_checkpoint.pt",
    )

    print(f"checkpoint saved to model_checkpoint.pt")
    print(f"baseline_mean={baseline_mean:.4f}, baseline_std={baseline_std:.4f}")
