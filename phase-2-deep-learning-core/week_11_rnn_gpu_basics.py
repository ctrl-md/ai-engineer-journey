"""
Week 11 -- RNNs/LSTMs/GRUs, GPU/systems basics.
The core RNN recurrence, worked by hand, plus PyTorch's built-in
LSTM/GRU layers. RNNs were kept at conceptual depth rather than a
full from-scratch build, per the curriculum's note that transformers
(Phase 3) have largely superseded them for most sequence tasks.
"""

import torch
import torch.nn as nn


def relu(z):
    if z <= 0:
        return 0
    return z


def rnn_step(x_t, h_prev, w_x, w_h, b):
    """One step of a plain RNN's recurrence: h_t = activation(w_x*x_t + w_h*h_prev + b)"""
    return relu(w_x * x_t + w_h * h_prev + b)


if __name__ == "__main__":
    # hand-worked example: sequence [2, 1, 3], w_x=0.5, w_h=0.3, b=0, h0=0
    x_seq = [2, 1, 3]
    w_x, w_h, b = 0.5, 0.3, 0
    h = 0
    for t, x_t in enumerate(x_seq, start=1):
        h = rnn_step(x_t, h, w_x, w_h, b)
        print(f"h{t} = {h}")

    # vanishing gradients through time: w_h multiplied repeatedly across steps
    for steps in [1, 5, 10, 20]:
        print(f"w_h^{steps} = {w_h**steps:.10f}")

    # PyTorch's built-in LSTM/GRU -- single lines, same pattern as nn.Linear
    lstm = nn.LSTM(input_size=1, hidden_size=8, batch_first=True)
    gru = nn.GRU(input_size=1, hidden_size=8, batch_first=True)
    seq = torch.randn(1, 3, 1)  # batch of 1, sequence length 3, 1 feature
    lstm_out, (h_n, c_n) = lstm(seq)
    gru_out, h_n_gru = gru(seq)
    print(f"LSTM output shape: {lstm_out.shape}")
    print(f"GRU output shape: {gru_out.shape}")
